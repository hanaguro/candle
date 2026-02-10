// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

// Rustの属性（Attribute）である #[cfg(...)] は、「その直後の1つのアイテム」に対してのみ有効
// #[cfg] と対象アイテムの間に空行やコメントがあっても、それは無視される

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;

use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

// clap の ValueEnum という機能は、これらを自動的にコマンドラインに適した形式
//（通常は小文字やケバブケース）に変換して受け付けてくれる
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V1,
    V2,
    V3,
    V31,
    V3Instruct,
    V31Instruct,
    V32_1b,
    V32_1bInstruct,
    V32_3b,
    V32_3bInstruct,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
    #[value(name = "SmoLM2-1.7B")]
    SmolLM2_1B,
    #[value(name = "SmoLM2-1.7B-Instruct")]
    SmolLM2_1BInstruct,
    #[value(name = "SmoLM2-360M")]
    SmolLM2_360M,
    #[value(name = "SmoLM2-360M-Instruct")]
    SmolLM2_360MInstruct,
    #[value(name = "SmoLM2-135M")]
    SmolLM2_135M,
    #[value(name = "SmoLM2-135M-Instruct")]
    SmolLM2_135MInstruct,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    // default_value_tの指定する値の型はその変数の型そのもの
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 10000)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    // default_valueの指定する値の型は文字列 (&str)
    /// The model size to use.
    #[arg(long, default_value = "v3")]
    which: Which,

    #[arg(long)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;
    // Google Chromeのデベロッパーツールを使って、AIモデルの動きをタイムライン表示するためのデータを作る
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    // guard 変数が main 関数の終わりで消滅（Drop）するときに、
    // 溜まっていた記録データをまとめて JSON ファイルに書き出す。
    // だから _guard という名前でメモリ上に保持し続けている。
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // 可能であればGPU（CUDAなど）を使い、ダメならCPUを使う。
    // --cpu フラグを付けずに実行すれば、基本的には「GPUを使おうとする」 という動きになる。
    let device = candle_examples::device(args.cpu)?;

    // as_deref(): これにより Option<String> を Option<&str> に変換している。
    // なぜこれが必要か？ Rust の match 文で "f16" のような文字列リテラルと直接比較するには、
    // 所有権を持った String ではなく、参照である &str である必要があるから。
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        // もしユーザーが "int8" や "pizza" など、対応していない文字列を入力した場合、
        // bail! マクロによって即座にエラーメッセージを返してプログラムを終了させる。
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };
    let (llama, tokenizer_filename, mut cache, config) = {
        let api = Api::new()?;
        // unwrap_or_else: 「もし None だった時だけ、この { ... } の中の処理を実行してね」 という命令。
        let model_id = args.model_id.unwrap_or_else(|| {
            let str = match args.which {
                Which::V1 => "Narsil/amall-7b",
                Which::V2 => "meta-llama/Llama-2-7b-hf",
                Which::V3 => "meta-llama/Meta-Llama-3-8B",
                Which::V3Instruct => "meta-llama/Meta-Llama-3-8B-Instruct",
                Which::V31 => "meta-llama/Llama-3.1-8B",
                Which::V31Instruct => "meta-llama/Llama-3.1-8B-Instruct",
                Which::V32_1b => "meta-llama/Llama-3.2-1B",
                Which::V32_1bInstruct => "meta-llama/Llama-3.2-1B-Instruct",
                Which::V32_3b => "meta-llama/Llama-3.2-3B",
                Which::V32_3bInstruct => "meta-llama/Llama-3.2-3B-Instruct",
                Which::Solar10_7B => "upstage/SOLAR-10.7B-v1.0",
                Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                Which::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M",
                Which::SmolLM2_135MInstruct => "HuggingFaceTB/SmolLM2-135M-Instruct",
                Which::SmolLM2_360M => "HuggingFaceTB/SmolLM2-360M",
                Which::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct",
                Which::SmolLM2_1B => "HuggingFaceTB/SmolLM2-1.7B",
                Which::SmolLM2_1BInstruct => "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            };
            str.to_string()
        });
        println!("loading the model weights from {model_id}");
        let revision = args.revision.unwrap_or("main".to_string());
        
        // Hugging Faceという「AIモデルの巨大な図書館」から、「どの棚（モデル）の、
        // どのバージョン（リビジョン）」を取り出すかを確定させる作業。
        // RepoType::Model: Hugging Faceには「モデル」の他に「データセット」や「スペース」もある。
        // ここでは「これはAIモデルのリポジトリですよ」と明示している
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        // キャッシュがあればそれを使い、なければインターネットからダウンロードする。
        // 「ローカルディスク上のどこにそのファイルがあるか」というパス情報（PathBuf）を返す。
        let tokenizer_filename = api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;

        // ファイルの読み込みと「デシリアライズ（構造体への変換）」を同時に行っている。
        // std::fs::read(config_filename)?: 指定されたパスにあるファイルを読み込み、
        // 「バイト列（Vec<u8>）」として取り出す。
        // serde_json::from_slice(...): Serde というRustの強力なライブラリを使い、
        // JSONという「ただの文字データ」を、LlamaConfig というRustの構造体に一瞬でマッピングする。
        // std::fs::read(path) は、ファイルの中身をテキストとしてではなく、生データ（Vec<u8>）として読み込む。
        // わざわざ一回「テキスト（String）」に変換してからパースするよりも、
        // 「バイト列のまま直接パース」する方がメモリも時間も節約できるため、
        // パフォーマンスが求められるAIプログラムではこちらが好まれる。
        // from_slice という関数自体は汎用的だが、左辺にLLamaConfig という構造体を指定することで、
        // 型推論でRustコンパイラがLlamaConfig という構造体にマッピングすることを判断する。
        // LlamaConfigは#[derive(Deserialize)]を付けて定義されている。
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;

        // 汎用的な設定から「LLaMA専用の最適化された設定」へ変換する。
        // into_config: Hugging Faceから取ってきた「素のデータ（LlamaConfig）」を、
        // AIモデル（Llama）が推論で使いやすい内部形式に変換する。
        // args.use_flash_attn: ユーザーがコマンドラインで指定した Flash Attention（フラッシュ・アテンション）を
        // 使うかどうかのフラグを渡している。
        let config = config.into_config(args.use_flash_attn);

        let filenames = match args.which {
            // 巨大モデル（分割ファイル）の読み込み。
            // model.safetensors.index.json: これは「どの重みが、どのファイル（01, 02...）に入っているか」
            // を記した目次ファイル。
            // hub_load_safetensors はこの目次を読み、必要なすべての分割ファイル
            // （.safetensors）を自動的にダウンロード・特定し、そのパスのリスト（Vec<PathBuf>）を返す。
            Which::V1
            | Which::V2
            | Which::V3
            | Which::V3Instruct
            | Which::V31
            | Which::V31Instruct
            | Which::V32_3b
            | Which::V32_3bInstruct
            | Which::Solar10_7B => {
                candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?
            }

            // 小規模モデル（単一ファイル）の読み込み。
            // 単一ファイル: これらはサイズが小さいため、重みが model.safetensors 
            // という1つのファイルにすべて収まっている。
            // api.get(...): そのファイルをダウンロードし、その場所（住所）を1つ返す。
            // vec![...]: 後の処理（VarBuilder）が「ファイルのリスト」を
            // 受け取る仕組みになっているため、1つしかファイルがなくても、
            // わざわざ Vec（リスト形式）に入れて形を整えている。
            Which::SmolLM2_360M
            | Which::SmolLM2_360MInstruct
            | Which::SmolLM2_135M
            | Which::SmolLM2_135MInstruct
            | Which::SmolLM2_1B
            | Which::SmolLM2_1BInstruct
            | Which::V32_1b
            | Which::V32_1bInstruct
            | Which::TinyLlama1_1BChat => {
                vec![api.get("model.safetensors")?]
            }
        };
        
        // 「推論を爆速にするためのメモ帳」であるKVキャッシュ（Key-Value Cache）を準備する。
        // 役割: 一度計算した「言葉のつながり」を保存し、再計算を防ぐ。
        // メリット: 生成速度が、文章の長さに関わらず常に高速に保たれる。
        let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;

        // VarBuilder は、いわば「重みの管理人」。
        // やっていること: 巨大な知能（重み）をメモリ上に「マッピング」する。
        // メリット: 起動がめちゃくちゃ速くなり、メモリ消費も最小限に抑えられる。
        // unsafe: パフォーマンスと引き換えに、低レイヤーの操作を行うための儀式。
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

        (Llama::load(vb, &config)?, tokenizer_filename, cache, config)
    };

    // E::msg: anyhow クレートの機能を使って、「どんな種類のエラーでも、とりあえずそのメッセージを
    // 保持した汎用的なエラーに変換する」という処理をしている。
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // AIが文章の生成を終わらせるための「終了の合図（EOS: End Of Sequence）」を特定する処理。
    // config.eos_token_idがなければ、辞書から直接探す。
    // or_else: 設定が空だったときの救済措置（フォールバック）。
    let eos_token_id = config.eos_token_id.or_else(|| {
        tokenizer
            // tokenizer.token_to_id(EOS_TOKEN): 辞書（tokenizer）に向かって、
            // 「</s>（文末を表す記号）」という文字に対応するIDは何番？と問い合わせる。
            .token_to_id(EOS_TOKEN)
            // .map(...): 無事にIDが見つかったら、それを LlamaEosToks::Single という型に包んで返す。
            .map(model::LlamaEosToks::Single)
    });

    // AIに渡す「最初の命令（プロンプト）」を決定するロジック。
    // as_ref(): Option<String> を Option<&String> に変換します。つまり、「中身をちょっと覗かせてね」
    // という参照の状態にする。
    // map_or(デフォルト値, クロージャ): None の場合は第1引数の DEFAULT_PROMPT（&str 型）をそのまま返し、
    // Some(s) の場合は第2引数のクロージャ |p| p.as_str() を実行する。&String を &str に変換して返す。
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());

    // 人間が読める「プロンプト（文字列）」を、AIが処理できる「トークンID（数値の配列）」に変換し、
    // さらに後で新しい単語を追加できるように準備する
    let mut tokens = tokenizer
        // テキストをトークン化します。
        // true の意味: 「特殊トークンを追加するかどうか」の設定。
        .encode(prompt, true)
        // 文字列に不正な文字が含まれているなどでトークン化に失敗した場合、
        // そのエラーを anyhow 型に変換して報告し、プログラムを安全に停止させる。
        .map_err(E::msg)?
        // encode メソッドは、単語の区切り位置（オフセット）など多くの情報を返すが、
        // AIの計算に必要なのは数値ID（[u32]）だけです。そのIDの配列（スライス）を取り出す。
        .get_ids()
        // get_ids() が返すのは「読み取り専用の参照」だが、
        // .to_vec() を使うことでメモリ上に新しく自分自身の Vec<u32>（可変長の配列）を作成する。
        .to_vec();

    // AIが生成した「数字（トークンID）」を、「文字」として画面にパラパラと表示させる際、
    // 文字化けを防ぎつつスムーズに処理するための重要なラッパー（包み役）を作成している。
    // 通常の tokenizer.decode() をそのまま使うと、
    // 日本語のようなマルチバイト文字（UTF-8）を扱う際に問題が起こる。
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    println!("starting the inference loop");
    print!("{prompt}");

    // AIが計算した膨大な「次の単語の候補リスト」から、実際にどの単語を口にするか（選択するか）を
    // 決める「意思決定エンジン」を構築する。
    let mut logits_processor = {
        // temperature: 低い（0に近い）と「最も確率が高いもの」を頑固に選ぶ。回答が正確になるが、単調になり、
        // 高い（1に近い）と確率が低いものも選ばれやすくなる。回答が創造的（クリエイティブ）になるが、
        // 支離滅裂になるリスクも増える。
        let temperature = args.temperature;
        // if temperature <= 0. という分岐は、「温度が0以下なら冒険は一切せず、
        // 常に1番確率が高いものを選ぶ（ArgMax）」という設定に切り替えている。
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                // 全単語を対象に、確率に基づいてランダムに選ぶ。
                (None, None) => Sampling::All { temperature },
                // 確率が高い順に K個 だけを候補に残し、それ以外を切り捨てる。
                (Some(k), None) => Sampling::TopK { k, temperature },
                // 確率の合計が P%（例: 90%）になるまでの上位単語だけを候補にする。
                (None, Some(p)) => Sampling::TopP { p, temperature },
                // まずK個に絞り、さらにその中で合計確率P%まで絞り込む。最も贅沢な絞り方。
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        // 決まった戦略（sampling）と args.seed（乱数の種）を使って、実際のプロセッサを作成する。
        // args.seed: 乱数の種を固定することで、同じプロンプトからは毎回同じ回答が出るように再現性を
        // 持たせることができる。
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..args.sample_len {
        // LLaMAの推論における「再計算の無駄を省くためのスイッチング」を行っている。
        // 最初（プロンプト）は全部まとめて計算するけれど、2単語目からは最新の1つだけを計算する。
        // AIが文章を作る際、前に計算した内容を覚えておく仕組み（KVキャッシュ）を使う。
        // このキャッシュのおかげで、毎回プロンプトの最初から計算し直す必要がなくなる。
        // context_size: 今回モデルに投入する単語数。
        // context_index: 投入する単語の「開始位置」。
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            // context_size = 1: 入力するのは「直前に生成された1単語」だけ。
            // context_index = index_pos: その1単語を、文章の「何番目」として
            // 処理するか（これまでの合計単語数）を指定する。
            (1, index_pos)
        } else {
            // context_size = tokens.len(): プロンプト全体の長さ分、すべてを計算対象にする。
            // context_index = 0: 文章の「0番目（最初）」から計算を開始する。
            (tokens.len(), 0)
        };
        if index == 1 {
            // 「プロンプトの読み込み時間」を除外し、「純粋な文字生成スピード」を正確に測定するため。
            start_gen = std::time::Instant::now()
        }

        // 現在持っているトークンの列（tokens）の中から、
        // 「今回、AIモデル（脳）に読み込ませる範囲」を切り出す処理。
        // .saturating_sub(context_size): 全個数から context_size を引く。
        // Rustでは0 - 1を行うとプログラムがクラッシュ（オーバーフロー）してしまう。
        // saturating_sub は、もし引き算の結果がマイナスになる場合は、
        // 安全に 0 で止めてくれる 魔法の引き算。
        // &tokens[開始位置..]: これは「スライス」という手法で、配列の特定の範囲を参照として取り出す。
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

        // Tensor::new(ctxt, &device)?: Rust標準のデータ（スライスやVec）を、
        // candleライブラリのTensor（多次元配列）オブジェクトに変換する。この時点の形は、
        // もし ctxt に5つのトークンが入っていたら、この時点での形は(5)
        // という1次元のベクトルになる。
        // .unsqueeze(0)?: unsqueeze(0) は、0番目（先頭）にサイズ1の新しい次元(バッチサイズ)を追加する。
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;

        // 順伝播（Forward Propagation）を行なう。
        // &input: 先ほど用意した「1人前の行列（テンソル）」。現在処理すべきトークンIDが含まれる。
        // context_index: 「今、文章全体の何番目を処理しているか」という位置情報。
        // &mut cache: KVキャッシュ。
        // 何をしている？: 入力データと過去の記憶（キャッシュ）を使い、AIに「次に来る言葉」を予想させている。
        // キャッシュの動き: 今回の知見を記憶に刻み込みつつ、古い記憶を取り出している。
        // 結果: 全単語に対する「次に来る確率の元（スコア）」が手に入る。
        let logits = llama.forward(&input, context_index, &mut cache)?;

        // squeeze（スクイーズ）は「絞り出す」「押しつぶす」という意味。
        // テンソルの次元の中で、サイズが「1」しかない余分な次元を取り除く操作をする。
        // Before (3次元): 形は (1, シーケンス長, ボキャブラリー数)。
        // After (2次元): 形は (シーケンス長, ボキャブラリー数)。
        // AIモデルは計算の都合上、常に「バッチ（束）」でデータを受け取り、バッチで返す。
        // しかし、この後の処理（どの単語を選ぶかの判定）では、「バッチ」という概念は邪魔になる。
        let logits = logits.squeeze(0)?;

        // AIが「同じ言葉を何度も繰り返してしまう現象（ループ）」を防ぐためのペナルティ処理を行っている。
        // AI（LLM）は放っておくと、「私は…私は…私は…」のように同じフレーズを無限に繰り返してしまう癖がある。
        // それを数学的に抑制するのがこのロジック。
        let logits = if args.repeat_penalty == 1. {
            // ペナルティの値が 1.0（デフォルト）の場合、計算をスキップしてそのままの logits を使う。
            // 数学的に「1を掛ける（または割る）」のは値を変えないため、重い計算を避けて高速化を図る。
            logits
        } else {
            // args.repeat_last_n: 「直近何単語分をチェックするか」という設定。
            // saturating_sub: 前にも出てきましたが、文章がまだ短くて last_n より少ない場合でも、
            // マイナスにならずに 0 で止まるようにしている。
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);

            // 履歴の確認: tokens[start_at..]（直近の履歴）に含まれている単語IDをリストアップする。
            // スコアの下落: それらの単語に対応する logits（スコア）を、args.repeat_penalty を
            // 使って強制的に下げる。直近の履歴に含まれる単語のスコアがプラスの場合、ペナルティ値で割る。
            // 直近の履歴に含まれる単語のスコアがマイナスの場合、ペナルティ値を掛ける（さらに低い点数に）。
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        // logits: 全単語の「期待度」が詰まった生のリスト。
        // sample(): 性格設定（温度やTop-P）を反映させつつ、確率に基づいたクジ引きを行う。
        // next_token: 当選した「たった一つの単語ID」。
        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        // さきほど決定した next_token（最新の1単語）が、事前に調べておいた
        // 「終了の合図（EOSトークン）」と一致するかどうかをチェックしている。
        // 一致した場合は break を実行し、無限に続く可能性がある生成ループを脱出する。
        match eos_token_id {
            // 合図が1つの場合（Single）
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                break;
            }
            // 合図が複数の場合（Multiple）
            // ref eos_ids: 複数のIDが入ったリスト（Vec）を、
            // 中身を壊さずに（所有権を奪わずに）「参照」として扱うために使われる。
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => (),
        }

        // tokenizer.next_token(next_token)?: 不完全な文字の待機、日本語の「あ」という文字が
        // 3つのトークン（バイト）に分かれていた場合、1つ目のトークンを入れただけでは、
        // この関数は None を返す。完成した瞬間に出力、3つ目のトークンが入って、
        // ようやく「あ」という文字が完成した瞬間に、Some("あ") を返す。
        // ? の役割: デコード中に予期せぬエラー（壊れたデータなど）が発生した場合、
        // すぐに呼び出し元へエラーを報告する。
        // next_token()の返り値はResult<Option<String>>なので、?はResultを開ける。
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            // コンピュータの標準出力は「効率化」のために、ある程度の文字数が溜まるか、
            // 改行（\n）が来るまで、画面に出さずにメモリ（バッファ）の中に溜めておくという性質がある。
            // flush() なしの場合: AIが裏でお喋りしていても画面には何も映らず、
            // 文章が全部完成した瞬間に「ドバッ」と一気に表示される。
            // flush() ありの場合: 「バッファに溜めずに、今すぐ画面に叩き出せ！」と強制命令を出す。
            std::io::stdout().flush()?;
        }
    }

    // デコーダーの口の中に残っている最後の文字を、残さず吐き出させる。
    // tokenizer.decode_rest(): TokenOutputStream の内部バッファ（待ち行列）を強制的に空にする。
    // もしバッファに未出力のデータがあれば、それを無理やり文字列（String）に変換して返す。
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
