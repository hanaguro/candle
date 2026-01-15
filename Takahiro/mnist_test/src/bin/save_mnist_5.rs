use candle_datasets;
use image::{ImageBuffer, Luma};

fn main() -> anyhow::Result<()> {
    let m = candle_datasets::vision::mnist::load()?;
    let test_images = m.test_images;
    let test_labels = m.test_labels;

    // 「5」を探す
    let test_labels_vec: Vec<u8> = test_labels.to_vec1()?;
    let mut target_index = 0;
    for (i, &label) in test_labels_vec.iter().enumerate() {
        if label == 5 {
            target_index = i;
            break;
        }
    }

    // データを取得
    let image_tensor = test_images.get(target_index)?;
    let image_vec: Vec<f32> = image_tensor.to_vec1()?;

    // ★診断：データの最大値を調べる
    let max_val = image_vec.iter().fold(0.0f32, |a, &b| a.max(b));
    println!("データの最大値: {}", max_val);

    // 画像保存
    let mut img_buf = ImageBuffer::new(28, 28);
    for (i, &pixel_val) in image_vec.iter().enumerate() {
        let x = (i % 28) as u32;
        let y = (i / 28) as u32;
        
        // 最大値が1.0付近なら、0.0-1.0のデータなので255倍する
        // 最大値が255付近なら、そのままでOK
        let val = if max_val <= 1.0 {
            (pixel_val * 255.0) as u8
        } else {
            pixel_val as u8
        };
        
        img_buf.put_pixel(x, y, Luma([val]));
    }

    let path = "real_5.png";
    img_buf.save(path)?;
    println!("画像を保存しました: {}", path);

    Ok(())
}
