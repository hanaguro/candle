// This should reach 91.5% accuracy.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};
use rand::prelude::*;
use rand::rng;

use candle::{DType, Result, Tensor, D};
use candle_nn::{loss, ops, Conv2d, Linear, Module, ModuleT, Optimizer, VarBuilder, VarMap};

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
    let bs = vs.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}

trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

struct LinearModel {
    linear: Linear,
}

impl Model for LinearModel {
    fn new(vs: VarBuilder) -> Result<Self> {
        let linear = linear_z(IMAGE_DIM, LABELS, vs)?;
        Ok(Self { linear })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}

struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Model for Mlp {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(IMAGE_DIM, 100, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, LABELS, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

#[derive(Debug)]
struct ConvNet {
    conv1: Conv2d,  // 第1畳み込み層【役割：原始的な特徴の抽出】
    conv2: Conv2d,  // 第2畳み込み層【役割：複雑なパターンの抽出】
    fc1: Linear,    // 第1全結合層【役割：抽出された特徴の整理・統合】
    fc2: Linear,    // 第2全結合層【役割：最終的な判定（0〜9の仕分け）】
    dropout: candle_nn::Dropout,    // ドロップアウト【役割：過学習の防止（カンニング防止）】
}

impl ConvNet {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        // スタート 入力画像    (初期値)    28x28
        // 第1段階  conv1       28-5+1      24x24
        // 第2段階  ブーリング  24/2        12x12
        // 第3段階  conv2       12-5+1      8x8
        // 第4段階  ブーリング  8/2         4x4
        // 
        // 64枚の特徴マップ(conv2の出力)
        // 各マップのサイズが4x4ピクセル
        // 合計：64x4x4=1024
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?; // b_sz: バッチサイズ、_img_dim: Image Dimension
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?    // 形を復元する(784 -> 28x28x1)
            .apply(&self.conv1)?    // 32種類のフィルターをかけ、画像から「エッジ(線)」などを抽出
            .max_pool2d(2)?         // 2x2の範囲から最大の値だけを取り出す
                                    // 画像の解像度をあえて落とし、重要な特徴だけを浮き彫りにする
                                    // 情報の間引き: 重要な特徴だけを残して、計算を軽くする
                                    // あそび（余裕）を作る: 多少のズレや形の崩れを許容できるようにする
            .apply(&self.conv2)?    // 抽出された「線」を組み合わせて、さらに複雑な「角」や「カーブ」を抽出(64種類)
            .max_pool2d(2)?         // 再び解像度を半分(8->4)にする
            .flatten_from(1)?       // (64, 4, 4)を、再び1列のデータ(1024個)に平らに並べ直す
            .apply(&self.fc1)?      // 抽出された 1024 個の特徴すべてを組み合わせて、新しい1024個の「判断材料」を作る
            .relu()?;               // ReLU(Rectified Linear Unit)関数を適用
                                    // 情報の整理: 不要なマイナス情報をカットし、重要なプラス情報だけを活かす
                                    // 知能の付与: 単純な足し算・掛け算の世界に「曲がり」を加え、複雑な判断を可能にする
                                    // 高速化: 計算が単純なので、大規模なモデルでもサクサク動く

        // forward_t():trainがtrueだと、学習中だけランダムに情報の伝達を遮断
        // apply():最終的に「0〜9の各数字に対するスコア」を算出
        self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
    }
}

struct TrainingArgs {
    learning_rate: f64,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
}

fn training_loop_cnn(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    const BSIZE: usize = 64;

    let dev = candle::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // VarMapは「畳み込みフィルター」の実体を保持する
    let mut varmap = VarMap::new();
    // VarBuilder は 「CNNという複雑な建物を建てるための、現場監督（インターフェース）」
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ConvNet::new(vs.clone())?;

    if let Some(load) = &args.load {
        println!("loading weights from {load}");
        varmap.load(load)?
    }

    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // train_images.dim(0)もBSIZEも整数なので、n_batchesには小数点以下が切り捨てられた数値が入る
    let n_batches = train_images.dim(0)? / BSIZE;   
    // 範囲0..n_batchesの最大値はn_batches-1
    // train_imagesの最後の方はトレーニングに使用されないことになる
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 1..=args.epochs {
        let mut sum_loss = 0f32;    // f32(32bit 浮動小数点数)の0
        batch_idxs.shuffle(&mut rng());
        for batch_idx in batch_idxs.iter() {
            // 巨大な多次元配列（Tensor）の中から、特定の範囲だけを「細長く」切り出す（スライスする）ための操作
            let train_images = train_images.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let train_labels = train_labels.narrow(0, batch_idx * BSIZE, BSIZE)?;

            // 入力された画像データがどの数字（0〜9）に近いか、モデルに推論（計算）させる
            let logits = model.forward(&train_images, true)?;

            // モデルが出力した「生のスコア（Logits）」を、
            // 「確率の対数（Log-Probability）」という、数学的に扱いやすい形に変換する
            // D::Minus1 は、一番最後の次元（Dimension）」に対して計算を行うという指定
            // データの形状: logits は [64, 10] という形をしている（64枚の画像 × 10個の数字スコア）
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;

            // モデルの予測結果（log_sm）と実際の正解（train_labels）を照らし合わせ、
            // 「モデルがどれだけ間違っていたか」を具体的な数値（損失値）として算出する処理
            let loss = loss::nll(&log_sm, &train_labels)?;

            // 算出された誤差（loss）を元に、モデル内の膨大なパラメーターを微調整して賢くする
            // 誤差逆伝播 (Backpropagation / 勾配の計算)とパラメータの更新 (Optimization Step)を行なっている
            opt.backward_step(&loss)?;

            // GPU（デバイス）上の計算結果であるTensorから数値を取り出し、
            // Rust標準の変数（CPU側）にコピーして足し合わせる
            // to_vec0::<f32>()は、0次元のTensor（スカラー値）を、
            // Rustのプリミティブ型（この場合は f32）に変換
            // to_vec0: 「0次元のベクトル（＝ただの数値）」としてデータを取り出すという意味
            sum_loss += loss.to_vec0::<f32>()?;
        }
        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward(&test_images, false)?;

        let sum_ok = test_logits
            .argmax(D::Minus1)?     // 各行（各画像）の中で、最もスコアが高かったインデックス（0〜9の番号）を返す
            .eq(&test_labels)?      // モデルの予測結果と、あらかじめ用意された正解ラベル（test_labels）を1つずつ比較
            .to_dtype(DType::F32)?  // 前のステップで得られた 0/1（通常は U8 型などの整数）を、f32（浮動小数点数）に変換
            .sum_all()?             // Tensor 内にあるすべての数値を足し合わせる
            .to_scalar::<f32>()?;   // GPU 上に存在する 0 次元の Tensor（ただの 1 つの数値）を、
                                    // Rust 標準の f32 型の変数として取り出し、CPU 側（メモリ）に転送

        // test_labels.dims1()?: テストデータのラベルが格納されているTensor（test_labels）の、
        // 1次元目の要素数（＝データの総数）を取得
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;

        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }
    if let Some(save) = &args.save {
        println!("saving trained weights in {save}");
        varmap.save(save)?
    }
    Ok(())
}

fn training_loop<M: Model>(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    let dev = candle::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = M::new(vs.clone())?;

    if let Some(load) = &args.load {
        println!("loading weights from {load}");
        varmap.load(load)?
    }

    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    for epoch in 1..=args.epochs {
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    if let Some(save) = &args.save {
        println!("saving trained weights in {save}");
        varmap.save(save)?
    }
    Ok(())
}

#[derive(ValueEnum, Clone)]
enum WhichModel {
    Linear,
    Mlp,
    Cnn,
}

#[derive(Parser)]
struct Args {
    #[clap(value_enum, default_value_t = WhichModel::Linear)]
    model: WhichModel,

    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 200)]
    epochs: usize,

    /// The file where to save the trained weights, in safetensors format.
    #[arg(long)]
    save: Option<String>,

    /// The file where to load the trained weights from, in safetensors format.
    #[arg(long)]
    load: Option<String>,

    /// The directory where to load the dataset from, in ubyte format.
    #[arg(long)]
    local_mnist: Option<String>,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Load the dataset
    let m = if let Some(directory) = args.local_mnist {
        candle_datasets::vision::mnist::load_dir(directory)?
    } else {
        candle_datasets::vision::mnist::load()?
    };
    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    let default_learning_rate = match args.model {
        WhichModel::Linear => 1.,
        WhichModel::Mlp => 0.05,
//        WhichModel::Cnn => 0.001,
        WhichModel::Cnn => 0.00005,
    };
    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
        load: args.load,
        save: args.save,
    };
    match args.model {
        WhichModel::Linear => training_loop::<LinearModel>(m, &training_args),
        WhichModel::Mlp => training_loop::<Mlp>(m, &training_args),
        WhichModel::Cnn => training_loop_cnn(m, &training_args),
    }
}
