use clap::Parser;
use candle_core::{DType, Result, Tensor};
use candle_nn::{VarBuilder, Conv2d, Linear, ModuleT};
use candle_datasets;

// ----------------------
// User's ConvNet Struct
// ----------------------
const LABELS: usize = 10;

struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: candle_nn::Dropout,
}

impl ConvNet {
    fn new(vs: VarBuilder) -> Result<Self> {
        // ★修正点: inspectの結果に合わせて "c1", "c2" に戻しました
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self { conv1, conv2, fc1, fc2, dropout })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
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
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_core::Device::cuda_if_available(0)?;

    println!("--- 診断開始 ---");
    println!("モデル読み込み: {}", args.model_path);

    // --- 修正ポイント：ここから ---
    // 1. inspect.rs と同じ方法で、ファイルを直接 HashMap として読み込む
    let weights = candle_core::safetensors::load(&args.model_path, &device)?;

    // 2. そのデータを使って VarBuilder を作成する
    // これにより、ファイルにある名前 (c1, c2, fc1, fc2) が強制的にモデルに紐付けられます
    let vs = VarBuilder::from_tensors(weights, DType::F32, &device);
    // --- 修正ポイント：ここまで ---

    // モデル作成 (vs.pp("c1") などがファイル内の名前に自動でマッチします)
    let model = ConvNet::new(vs)?;

    // 2. データセットのロード (正規化: 0.0 ~ 1.0)
    let m = candle_datasets::vision::mnist::load()?;
    let test_images = m.test_images.to_device(&device)?;

    // インデックス 8 番の「5」を取得
    let target_idx = 8; 
    let input = test_images.get(target_idx)?.unsqueeze(0)?;

    // 推論実行
    let logits = model.forward(&input, false)?;
    let probabilities = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
    let prob_vec = probabilities.to_vec2::<f32>()?[0].clone();
    
    let mut indexed_probs: Vec<(usize, f32)> = prob_vec.into_iter().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n--- 判定結果 (データセット直接入力) ---");
    for i in 0..3 {
        println!("第{}位: 数字 {} (確信度: {:.2}%)", i + 1, indexed_probs[i].0, indexed_probs[i].1 * 100.0);
    }

    Ok(())
}

