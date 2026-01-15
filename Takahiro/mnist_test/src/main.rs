use clap::Parser;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, Conv2d, Linear, ModuleT, Dropout};
use image::io::Reader as ImageReader;

const LABELS: usize = 10;

struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl ConvNet {
    fn new(vs: VarBuilder) -> candle_core::Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self { conv1, conv2, fc1, fc2, dropout })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
    }
}

#[derive(Parser)]
struct Args {
    model_path: String,
    img_name: String,
}

fn load_image28x28(path: &str, device: &Device) -> anyhow::Result<Tensor> {
    // .decode() はpng等の圧縮を解き、DynamicImage という型に変換
    let img = ImageReader::open(path)?.decode()?;
    // 8ビット（0〜255）のグレースケール（白黒）画像に変換
    let img = img.to_luma8();
    // 「28x28ピクセル」という規格サイズに縮小（または拡大）する処理
    let resized = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Lanczos3);

    // 診断結果が1.0だったので、ここでも255で割って0.0-1.0に正規化します
    let pixels: Vec<f32> = resized
        // 28x28にリサイズされた画像から、ピクセルを1つずつ取り出す「行列」（イテレータ）を作る
        .pixels() 
        // 数値の正規化（Normalization）を行なう
        .map(|p| p[0] as f32 / 255.0)   // Iteratorトレイトのメソッド
        // 加工された784個の数値を、バラバラな状態から1つの**「ベクトル（Vec<f32>）」**というリストにまとめる
        .collect();

    Ok(Tensor::from_vec(pixels, (1, 784), device)?)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;

    /* --- 失敗した「空の変数管理マップからロード方式」 ---
    // 1. 空の変数管理マップを作る
    let mut varmap = VarMap::new();
    // 2. ファイルからロードしようとする（ここで問題発生！）
    varmap.load(&args.model_path)?; 
    // 3. VarBuilder を作成
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    */

    // --- 成功した「直接ロード方式」 ---
    let weights = candle_core::safetensors::load(&args.model_path, &device)?;
    let vs = VarBuilder::from_tensors(weights, DType::F32, &device);

    let model = ConvNet::new(vs)?;

    // 画像の読み込み
    let input = load_image28x28(&args.img_name, &device)?;

    // 推論
    // 計算の結果、10個の出口（0〜9）から出てきた「生のスコア」
    let logits = model.forward(&input, false)?;
    // 10個の出口から出た数値をすべてプラスの値に変換し、さらに 「すべての合計がぴったり 1.0 (100%)」 になるように調整
    // D::Minus1: 「一番最後の次元（10個の数字が並んでいる方向）」に対してソフトマックスを計算せよ、という指示
    let probabilities = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
    // メインメモリ（CPU側）にデータを取り戻す
    let prob_vec = probabilities.to_vec2::<f32>()?[0].clone();

    // 「背番号（インデックス）」と「スコア（確率）」をセットにする工程
    let mut indexed_probs: Vec<(usize, f32)> = prob_vec     // prob_vecはただの数値の並び [0.01, 0.02, 0.10, 0.77, ...]
        .into_iter()    // リスト（Vec）を、1つずつ取り出せる「ベルトコンベア（イテレータ）」に乗せる
        .enumerate()    // 流れてくるデータに、**「0番から順番に番号札」**を付ける
                        // データが (インデックス, 確率) というペア（タプル）に変わる
        .collect();     // 番号札が付いたペアを、再び1つの新しいリスト（Vec）にまとめ上げる

    // ラベル付きのリストを「確率が高い順（降順）」に並び替える処理
    // partial_cmp: Rustで小数を比較するためのメソッド
    // b を先に書いて b.partial_cmp(&a) と書くことで、
    // 「bの方が大きいなら、bを前に持ってきてね」という命令になり、結果として大きい順になる
    // partial_cmp は Option 型を返す
    // 今回の確率は softmax を通った正常な数値（0.0〜1.0）であり、絶対に NaN
    // は含まれないので、unwrap()でRustに保証を与える
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("--- 判定結果 ---");
    for i in 0..10 {
        println!("第{}位: 数字 {} (確信度: {:.2}%)", i + 1, indexed_probs[i].0, indexed_probs[i].1 * 100.0);
    }

    Ok(())
}
