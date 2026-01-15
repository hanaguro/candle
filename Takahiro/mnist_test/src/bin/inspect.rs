use clap::Parser;
use candle_core::safetensors::load;
use candle_core::Device;

#[derive(Parser)]
struct Args {
    model_path: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;

    println!("ファイル検査開始: {}", args.model_path);
    
    // VarMapを使わず、低レベルな関数で直接読み込む
    // これでエラーが出るならファイルが壊れています
    let tensors = load(&args.model_path, &device)?;

    println!("--- 検出されたテンソル一覧 ---");
    if tensors.is_empty() {
        println!("警告: ファイルは読み込めましたが、中にテンソルが1つもありません！");
    } else {
        for (name, tensor) in tensors.iter() {
            println!("名前: {:<20} | 形状: {:?}", name, tensor.shape());
        }
    }
    println!("------------------------------");

    Ok(())
}
