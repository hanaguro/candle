#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::{Error as E, Result};
use candle::Tensor;
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation.
    #[arg(long, default_value = "false")]
    approximate_gelu: bool,

    /// Include padding token embeddings when performing mean pooling. By default, these are masked away.
    #[arg(long, default_value = "false")]
    include_padding_embeddings: bool,
}

impl Args {
    // このResultはanyhow::Result。Result<(BertModel, Tokenizer), E>というように、エラー側の型を書かなくても良い。
    fn build_model_and_tokenizer(&self) -> Result<(BertModel, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
        
        // self.model_idとself.revisionはOption<String>型
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            // Some(model_id) は、「Some かどうかをチェックし、同時に中身を model_id という名前で取り出す」 という構文
            (Some(model_id), Some(revision)) => (model_id, revision),
            // 左側の (Some(model_id), None) という記述自体が、新しい変数(model_id)の宣言を兼ねている
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        // repo 変数は、「どこに、何を取りに行くか」という情報をまとめただけのオブジェクト
        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            // 通信クライアント作成
            let api = Api::new()?;
            // repoを使うように指示
            let api = api.repo(repo);
            // 実際にダウンロード開始
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;

        // Tokenizer（トークナイザ）は、人間が話す「言葉」を、
        // コンピュータが計算できる「数字の列」に変換するための専用の翻訳機（辞書）
        // .map_err(E::msg) は、Rustのエラー処理において「エラーの型を、
        // 関数の戻り値に合うように変換する」という非常に重要な役割
        // map_err は、「もし結果が Err だったら、この関数（E::msg）を適用してエラーを作り直せ」という命令
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // AIの「知能」の実体である「重み（パラメータ）」データをファイルから読み込み、
        // モデルを組み立てるための準備をしている
        let vb = if self.use_pth {
            // PyTorchの標準形式の重みを読み込む
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            // Hugging Faceが開発した最新形式。高速で、セキュリティ的に安全な形式の重みを読み込む
            // ファイル名のリストの参照にしているのは、
            // AIモデルの重みデータが複数のファイルに分割されている場合（シャッディング）にも
            // 対応できるようにするため
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        if self.approximate_gelu {
            // 精密なGELU計算を捨てて、「近似式」を使って計算
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        let model = BertModel::load(vb, &config)?;
        Ok((model, tokenizer))
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {  // if式の内部のguardの寿命がmain()の最後まで伸びる
                                    // 「使わないけれど、消えてほしくない（メモリに居続けてほしい）」 変数には、
                                    // 慣習として _（アンダースコア）を頭に付ける
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();       // 追跡システムをセットアップ
        Some(guard)
    } else {
        None
    };
    let start = std::time::Instant::now();

    // インターネット上のデータから、PC内に『AIの推論エンジン』を組み立てる
    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.device;

    if let Some(prompt) = args.prompt {
        let tokenizer = tokenizer
            // パディングの無効化
            // 今回のコードは、1つの文章を処理したり、後続の処理で動的に長さを計算したりするため、
            // トークナイザ側で固定の長さに揃えてしまうと、計算に無駄が出たり、
            // 意図しない数値（0など）が混ざったりするのを防ぎたいから
            .with_padding(None)
            // 切り捨ての無効化
            // 今回のようなサンプル実行では、入力が短いことが分かっているため、自動で切る設定は不要
            .with_truncation(None)
            // .with_padding() などのメソッドは、設定に失敗すると tokenizers ライブラリ独自のエラーを返す
            // それを anyhow というこのプログラム共通のエラー形式（E::msg）に変換して、
            // ? で呼び出し元に投げられるようにする
            .map_err(E::msg)?;
        let tokens = tokenizer
            // テキストを解析し、単語を分割してIDを割り当てる
            // true (add_special_tokens): true にすることで、BERTに必要な特殊トークンを自動的に付与する
            // 戻り値: Result<Encoding, エラー型>が返される
            .encode(prompt, true)
            // map_err(): 処理中にエラーが発生した場合、そのエラーを anyhow 形式に変換する
            // ?: Result型からEncode構造体を取り出す、エラーの場合は関数を中断してエラーを返す
            .map_err(E::msg)?
            // Encoding 構造体から、計算に一番必要な 「単語ID（数値）」のリストだけを取り出す
            // スライス（&[u32]）が返される
            .get_ids()
            // スライス（&[u32]）を新しくコピーして、Vec<u32> というRustの標準的なベクトル形式に変換
            .to_vec();

        // Rustの標準的な数値リスト（Vec）を、「GPUで計算可能な多次元行列（Tensor）」へと変換し、
        // さらにAIモデルが要求する「形」に整える
        // [..] の意味: 「最初から最後まで全部」という範囲指定
        // &tokens とだけ書いても多くの場合動くが、
        // [..] と書くことで「このデータの全範囲をスライスとして扱います」という意図がより明確になる
        // Tensor::new(): Rustの標準的な数値リストを、「AI計算用のテンソル」に変換する
        // unsqueeze(): 指定した位置に「サイズ1の新しい次元」を挿入する
        // unsqueeze(): 実行前：[7] → 実行後：[1, 7] （「7つの数字の塊」が「1つ」ある状態）
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;

        // zeros_like() は、「指定したテンソルと同じ形（Shape）、同じ型（DType）、
        // 同じデバイス（GPU/CPU）で、中身がすべて『0』のテンソルを新しく作る」というメソッド
        let token_type_ids = token_ids.zeros_like()?;

        println!("Loaded and encoded {:?}", start.elapsed());
        for idx in 0..args.n {
            let start = std::time::Instant::now();

            // 順伝播
            // &token_ids: [1, 7] の形のテンソル
            // &token_ids: AIに「どの単語（ID）が並んでいるか」を伝える
            // &token_type_ids: 「全部 0」のテンソル
            // &token_type_ids: AIに「これはすべて1番目の文章のデータですよ」と教える
            // None: 本来ここは「どのトークンを無視するか」を指定する場所
            // None: 「すべてのトークン（7個すべて）を無視せずに計算してください」という意味
            // 成功すると、計算結果が Tensor 型で返ってくる
            let ys = model.forward(&token_ids, &token_type_ids, None)?;
            if idx == 0 {
                println!("{ys}");
            }
            println!("Took {:?}", start.elapsed());
        }
    } else {
        let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];
        let n_sentences = sentences.len();

        // 複数の文章を一度に処理（バッチ処理）するために、「文章の長さを、
        // そのバッチ内の一番長い文に合わせて自動的に揃える」という設定を行なう
        // get_padding_mut(): トークナイザが現在持っているパディング設定を「書き換え可能な状態（可変参照）」で
        // 取り出す
        // get_padding_mut(): Option<&mut PaddingParams>を返す
        // if let は、「中身があるかのチェック」と「中身の取り出し（アンラップ）」を同時に行うのが最大の特徴
        if let Some(pp) = tokenizer.get_padding_mut() {
            // もし設定が存在すれば: 新しく作り直すのではなく、
            // その中の strategy という項目だけを BatchLongest に上書きする
            // BatchLongest: バッチ内の最長に合わせる
            // pp: &mut PaddingParams型
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            // もし設定がなければ: 新しく PaddingParams という設定オブジェクトを作成する
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            // with_padding(Some(pp)): 作成した設定をトークナイザにセットする
            // Some(pp): ここが「Option という箱」に入れている瞬間
            tokenizer.with_padding(Some(pp));
        }

        // 「パディング（長さを揃える）ルール」を適用しながら、
        // 8つの文章を一気に「数字のリストの束」へと変換する処理
        let tokens = tokenizer
            // encode が1つの文章を処理するのに対し、encode_batch は複数の文章を同時に処理する
            // to_vec(): [&str; 8] を Vec<&str> に変換
            // true (add_special_tokens): true にすることで、BERTに必要な特殊トークンを自動的に付与する
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;
        
        // 8個の文章それぞれの「IDリスト（Rustの普通の配列）」を、
        // 「8個の独立したテンソル（GPUで計算できる形式）」へ1つずつ変換し、
        // 最後にそれを1つのリストにまとめ直すという作業を行う
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                // tokens.get_ids().to_vec(): 1つの文章のIDリストを取り出す
                let tokens = tokens.get_ids().to_vec();
                // Tensor::new(..., device)?: そのIDリストを、指定されたデバイス（GPUなど）のメモリに送り、テンソル化する
                // Ok(...): 成功したテンソルを Result という箱に包んで返す
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            // Result<Vec<_>> という型を指定して collect すると、
            // 「もし全部成功していたら、中身をまとめて Ok(Vec<Tensor>) にし、
            // 1つでも失敗していたら、その時点でエラーとして処理する」 という動きになる
            // 最後の ?: これにより、無事に全部成功したあとの Vec<Tensor>（8個のテンソルが入ったリスト） 
            // だけが変数 token_ids に代入される
            // Result<Vec<_>> の _（アンダースコア）は、
            // Rustにおける 「型推論（型をコンパイラにお任せする）」ためのプレースホルダー（身代わり）
            .collect::<Result<Vec<_>>>()?;

        // 「AIに、文章のどこが本物の単語で、どこが単なる埋め草（パディング）かを教えるための
        // 『目隠し（マスク）』データ」をGPU上に準備している
        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                // .get_attention_mask(): Encoding 構造体の中から、
                // [1, 1, 1, 0, 0] といった 0 と 1 のリストを取り出す
                // .to_vec(): それを Rust の標準的な配列（Vec）にコピーする
                let tokens = tokens.get_attention_mask().to_vec();
                // 0 と 1 のリストを GPU（またはCPU）のメモリへ転送し、
                // AIが計算に使える「テンソル」という形式に変換する
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            // 8つの文章分、この作業を繰り返して、最終的に「8個のマスク・テンソル」が入ったリストを作る
            .collect::<Result<Vec<_>>>()?;

        // 「8つのバラバラな文章（1次元データ）を、
        // 1つの巨大な行列（2次元データ）に積み上げる」という非常に重要な作業を行なう
        // 0 を指定: 一番外側に新しい次元を作る。結果、[8, 7]という形になる
        // token_idsの参照を渡しているが左辺も同じ名前にしているため、以後合体した後の1つのテンソルになる
        let token_ids = Tensor::stack(&token_ids, 0)?;

        let attention_mask = Tensor::stack(&attention_mask, 0)?;

        // token_ids（文章のID行列）と全く同じ形、同じデータ形式、同じ場所（GPUなど）で、
        // 中身がすべて 「0」 のデータを作る
        let token_type_ids = token_ids.zeros_like()?;

        println!("running inference on batch {:?}", token_ids.shape());

        // 8つの文章（バッチ）と、それに対応するマスクデータを一気に流し込み、「意味の塊（ベクトル）」を生成する
        // 計算が終わって返ってくる embeddings は、[8, 7, 384] という3次元の形をした巨大なテンソル
        // 8: バッチサイズ。8つの文章それぞれの計算結果が入っている。
        // 7: シーケンス長。1つの文章につき、7つのトークン（単語）分の結果がある。
        // 384: 隠れ層の次元数。1つの単語につき、384個の数字でその意味が表現されている。
        let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        println!("generated embeddings {:?}", embeddings.shape());

        // 「単語ごとのベクトル（7個）」をギュッとまとめて、
        // その文章の代表値である「文章全体のベクトル（1個）」に変換する処理を行う。
        // この if-else ブロックを抜けると、embeddings の形は [8, 384] になる。
        // 8: 8つの文章
        // 384: 各文章の「意味」を凝縮したベクトル
        let embeddings = if args.include_padding_embeddings {
            // Apply avg-pooling by taking the mean embedding value for all
            // tokens, including padding. This was the original behavior of this
            // example, and we'd like to preserve it for posterity.

            // やり方: 7個のベクトルを全部足して、単純に 7 で割る。
            // 問題点: 7個の中には意味のない [PAD]（詰め物）も含まれている。
            // これらも平均に含めてしまうと、文章の本当の意味が薄まって（薄汚れて）しまう。
            // dims3() メソッド: 「このテンソルは3次元のはずだ」という前提で、
            // その3つの数字（次元）を (usize, usize, usize) というタプル形式でまとめて返す。
            let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;

            // 全単語の合計値を、この n_tokens で割ることで、
            // 「単語数で割った平均値」を算出しようとしている。
            // そのための「分母」となる数字を、dims3() で安全に取り出していた。
            // sum(1) の引数である 「1」 は、テンソルのどの方向（次元）に向かって数字を
            // 足し合わせるかという 「次元の番号（ディメンション・インデックス）」 を指定。
            // sum(1) を実行すると、その次元が「合計」されて消えるため、テンソルの形が以下のように変わる。
            // 計算前: [8, 7, 384]
            // 計算後: [8, 384]
            // その後に/ n_tokens（7で割る）をしているので、
            // 最終的に 「8つの文章それぞれの、単語の平均ベクトル」 が得られる。
            (embeddings.sum(1)? / (n_tokens as f64))?
        } else {
            // Apply avg-pooling by taking the mean embedding value for all
            // tokens (after applying the attention mask from tokenization).
            // This should produce the same numeric result as the
            // `sentence_transformers` Python library.

            // 「本物の単語だけ」を使って平均を計算する、より高度で正確な方法。

            // マスク（0か1が入った[8, 7]のデータ）を、計算できるように浮動小数点に変換し、
            // 形を[8, 7, 1]に拡張する。
            // unsqueeze(2): 次元の追加。
            // 操作前: 形（Shape）は [8, 7]。8つの文章, 7つの単語。
            // 操作後: 形は [8, 7, 1]。8つの文章, 7つの単語, 1つの箱。
            let attention_mask_for_pooling = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;

            // 1次元目（単語の方向）を合計する。
            // 結果は[8, 1]。各文章に「本物の単語が何個あるか（例：5個、7個、3個…）」というリストになる。
            let sum_mask = attention_mask_for_pooling.sum(1)?;

            // embeddings（元のデータ）にマスクを掛け算する。
            // これにより、[PAD] の部分の数値がすべて 0 になる。その上で、全単語を合計する。
            // embeddings.broadcast_mul(&attention_mask_for_pooling)について。
            // やっていること: 384次元のベクトル1つ1つに対して、マスクの値（1.0 または 0.0）を掛け算する。
            // ブロードキャストの魔法: マスクは[8, 7, 1]という形をしている。
            // 最後の「1」が、embeddings の「384」に合わせて自動的に引き延ばされる。
            let embeddings = (embeddings.broadcast_mul(&attention_mask_for_pooling)?).sum(1)?;

            // 「パディングを0にした合計値」を、「本物の単語数」で割る。
            // 純粋に意味のある単語だけを平均した、綺麗な 384 次元のベクトルが手に入る。
            // 通常、行列の割り算は形が一致していないとできないが、
            // broadcast_div は「サイズ 1 の次元」を自動的に引き伸ばして計算してくれる。
            // この一行が完了した瞬間、変数 embeddings は以下の状態になる。
            // 形: [8, 384]
            // 意味: 8 つの文章それぞれの、文脈を考慮した「純粋な意味」を凝縮したベクトル。
            embeddings.broadcast_div(&sum_mask)?
        };

        // この分岐は、ユーザーが「検索精度を上げたいか、生の数値が欲しいか」を選べるようにしている。
        let embeddings = if args.normalize_embeddings {
            // 正規化するメリット: 前述の通り、「内積（掛け算）＝ 類似度」 になるため、
            // 検索システム（RAGなど）を作る場合は、ほぼ確実にここを true にする。
            normalize_l2(&embeddings)?
        } else {
            // 正規化しない場合: 文章の「強さ」や「情報の量」がベクトルの長さに残る。
            // 特定の特殊な分析（クラスタリングの性質を細かく見たい場合など）では、
            // あえてそのままにすることもある。
            embeddings
        };
        println!("pooled embeddings {:?}", embeddings.shape());

        let mut similarities = vec![];

        // 外側のループ (i): 比較の基準となる文章を選ぶ。
        // 内側のループ (j): 相手となる文章を選ぶ。
        for i in 0..n_sentences {
            let e_i = embeddings.get(i)?;
            // i + 1 の意味: 自分自身との比較（i == j）や、
            // 既に計算したペア（AとB を計算した後の BとA）をスキップするための工夫。
            for j in (i + 1)..n_sentences {
                let e_j = embeddings.get(j)?;

                // コサイン類似度の計算ステップ
                // 内積（分子）: sum_ij
                let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
                // ベクトルの強さ（分母）: sum_i2, sum_j2
                let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
                let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;

                // 内積を「長さの掛け算」で割る。この計算により、結果は必ず -1.0 から 1.0 の間に収まる。
                // 1.0に近い: 意味が非常に似ている。
                // 0.0に近い: 全く関係がない。
                // 数学（幾何学）のルールとしては -1.0 まであり得るが、
                // AI（自然言語処理）の実用上は 0.0 から 1.0 の間に収まることがほとんど。
                // もし、このループの前に normalize_l2（正規化）を行っていた場合、
                // 実は sum_i2 と sum_j2 は必ず 1.0 になる（長さが1に揃えられているため）。
                let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                similarities.push((cosine_similarity, i, j))
            }
        }

        // sort_by(): 通常、整数のリストなどは .sort() だけで並び替えらるが、
        // 今回は「スコア、文章番号i、文章番号j」というタプルをソートしたいので、
        // どの値を使って比較するかを指定する sort_by を使う。
        // total_cmp: 小数の世界には NaN（Not a Number：計算不能な値）という特殊な値が存在する。
        // total_cmp は、そんな厄介な NaN や -0.0、+Infinity なども含めて、
        // 「無理やりでも一列に並べるためのルール」を提供してくれるメソッド。
        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));

        for &(score, i, j) in similarities[..].iter() {
            println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
        }
    }
    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    // v.sqr(): 各要素を2乗する。長さを計算するための準備（ピタゴラスの定理のa^2 + b^2の部分）。
    // sum_keepdim(1): 1番目の次元（384次元の方向）を合計する。形が[8, 384]から[8, 1]になる。
    // もしsum(1)を使うと、形は [8]（ただの8個の数字の列）になってしまう。
    // sqrt(): 合計値の平方根を取る。これでベクトルの「現在の長さ（ノルム）」が求まった。
    // broadcast_div(...): 「元のベクトル」を「現在の長さ」で割る。どんな長さのベクトルも、
    // 自分自身の長さで割れば、長さは必ず 1.0 になる。
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
