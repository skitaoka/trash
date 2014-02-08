//------------------------------------------------------------------------------
//
// BMP関係
//
//------------------------------------------------------------------------------
/// 
/// ファイルヘッダ(14 byte)
/// 
struct BITMAPFILEHEADER
{
  unsigned short  type;       ///< ファイルタイプ(BM)
  unsigned int    size;       ///< ファイルサイズ
  unsigned short  reserved1;  ///< 予約領域(0)
  unsigned short  reserved2;  ///< 予約領域(0)
  unsigned int    offBits;    ///< ファイル先頭から画像データまでのオフセット
};

///  
/// OS/2情報ヘッダ(12 byte)
/// 
struct BITMAPCOREHEADER
{
  unsigned int    size;     ///< 情報ヘッダのサイズ(12)
  unsigned short  width;    ///< 画像の幅
  unsigned short  height;   ///< 画像の高さ(正数なら画像データは下から上へ、負数なら画像データは上から下へ)
  unsigned short  planes;   ///< プレーン数(1)
  unsigned short  bitCount; ///< 1 画素あたりのデータサイズ
                            ///<   1 - 2 色ビットマップ
                            ///<   4 - 16 色ビットマップ
                            ///<   8 - 256 色ビットマップ
                            ///<   (16 - 65536色(high color)ビットマップ 正式に対応していない)
                            ///<   24 - 1677万色(true color)ビットマップ
                            ///<   32 - 1677万色(true color)ビットマップ
};

/// 
/// Windows情報ヘッダ(40 byte)
/// 
struct BITMAPINFOHEADER
{
  unsigned int	  size;           ///< 情報ヘッダのサイズ(40)
  int             width;          ///< 画像の幅
  int             height;         ///< 画像の高さ
                                  ///<   正数なら画像データは下から上へ
                                  ///<   負数なら画像データは上から下へ
  unsigned short  planes;         ///< プレーン数(1)
  unsigned short  bitCount;       ///< 1 画素あたりのデータサイズ
                                  ///<   1 - 2 色ビットマップ
                                  ///<   4 - 16 色ビットマップ
                                  ///<   8 - 256 色ビットマップ
                                  ///<   (16 - 65536色(high color)ビットマップ 正式に対応していない)
                                  ///<   24 - 1677万色(true color)ビットマップ
                                  ///<   32 - 1677万色(true color)ビットマップ
  unsigned int    compression;    ///< 圧縮形式
                                  ///<   0 - BI_RGB (無圧縮)
                                  ///<   1 - BI_RLE8 (RunLength 8 bits/pixel)
                                  ///<   2 - BI_RLE4 (RunLength 4 bits/pixel)
                                  ///<   3 - Bitfields
  unsigned int    sizeImage;      ///< 画像データ部のサイズ、0 の場合もある
  int             xPelsPerMeter;  ///< 横方向解像度(1mあたりの画素数)、0 の場合もある
  int             yPelsPerMeter;  ///< 縦方向解像度(1mあたりの画素数)、0 の場合もある
  unsigned int    clrUsed;        ///< 格納されているパレット数(使用色数)、0 の場合もある
  unsigned int    clrImportant;   ///< 重要なパレットのインデックス、0 の場合もある
};

/// 
/// OS/2カラーパレット
/// 
struct RGBTRIPLE
{
  unsigned char blue;   ///< 青
  unsigned char green;  ///< 緑
  unsigned char red;    ///< 赤
};

/// 
/// Windowsカラーパレット
/// 
struct RGBQUAD
{
  unsigned char blue;     ///< 青
  unsigned char green;    ///< 緑
  unsigned char red;      ///< 赤
  unsigned char reserved; ///< 予約領域
};

//------------------------------------------------------------------------------
//
// WAV関係
//
//------------------------------------------------------------------------------
/// 
/// 全ての"fmt "が持っているデータ(14 byte)
/// 
struct WAVEFORMAT
{
  unsigned short  formatTag;      ///<フォーマットID
                                  ///<   リニアPCMならば1(01 00)
  unsigned short  channels;       ///< チャンネル数
                                  ///<   モノラルならば1(01 00)
                                  ///<   ステレオならば2(02 00)
  unsigned int    samplesPerSec;  ///< サンプリングレート
                                  ///<   44.1[kHz]ならば44100(44 AC 00 00)
  unsigned int    avgBytesPerSec; ///< データ速度[Byte/sec]
                                  ///<   44.1[kHz] 16[bit]ステレオ ならば
                                  ///<   44100*2*2 = 176400(10 B1 02 00)
  unsigned short  blockAlign;     ///< ブロックサイズ[チャンネル数*Byte/sample]
                                  ///<   16[bit]ステレオならば
                                  ///<   2*2 = 4(04 00)
};

/// 
/// PCM用の"fmt "(14 byte)
/// 
struct PCMWAVEFORMAT
{
	WAVEFORMAT      wf;
	unsigned short  bitsPerSample;  ///< サンプルあたりのビット数[bit/sample]
                                  ///<   WAVフォーマットでは8[bit]か16[bit]
};

/// 
/// 一般系の"fmt "(18 byte)
/// 
struct WAVEFORMATEX
{
	unsigned short  formatTag;      ///< フォーマットID
                                  ///<   リニアPCMならば1(01 00)
	unsigned short  channels;       ///< チャンネル数
                                  ///<   モノラルならば1(01 00)
                                  ///<   ステレオならば2(02 00)
	unsigned int    samplesPerSec;  ///< サンプリングレート、44.1[kHz]ならば44100(44 AC 00 00)
	unsigned int    avgBytesPerSec; ///< データ速度[Byte/sec]
                                  ///<   44.1[kHz] 16[bit]ステレオ ならば
                                  ///<   44100*2*2 = 176400(10 B1 02 00)
	unsigned short	blockAlign;     ///< ブロックサイズ[チャンネル数*Byte/sample]
                                  ///<   16[bit]ステレオならば
                                  ///<   2*2 = 4(04 00)
	unsigned short	bitsPerSample;  ///< サンプルあたりのビット数[bit/sample]
                                  ///<   WAVフォーマットでは8[bit]か16[bit]
	unsigned short	size;           ///< 拡張部分のサイズ
                                  ///<   *リニアPCMならば存在しない*
};

//------------------------------------------------------------------------------
//
// AVI関係
//
//------------------------------------------------------------------------------
/// The AVI File Header LIST chunk should be padded to this size
#define AVI_HEADERSIZE 2048 // size of AVI header list

struct MainAVIHeader
{
  unsigned int microSecPerFrame;    ///< フレーム間の間隔(マイクロ秒単位)
  unsigned int maxBytesPerSec;      ///< ファイルの概算最大データレート
  unsigned int reserved1;           ///< 予約領域(0)
  unsigned int flags;               ///< ファイルに対するフラグ
                                    ///<   0x00010 - AVIF_HASINDEX (ファイルが"idx1"チャンクを含んでいる)
                                    ///<   0x00020 - AVIF_MUSTUSEINDEX (インデックスがフレームの順序図家のために使われている)
                                    ///<   0x00100 - AVIF_ISINTERLEAVED (インターリーブ付き)
                                    ///<   0x10000 - AVIF_WASCAPTUREFILE (ビデオをとるのに使われた)
                                    ///<   0x20000 - AVIF_COPYRIGHTED (著作権つきでコピー不可)
  unsigned int totalFrames;         ///< ファイル内のデータのフレームの総数
  unsigned int initialFrames;       ///< インターリーブされたファイルの開始フレーム
  unsigned int streams;             ///< ファイル内のストリーム数
  unsigned int suggestedBufferSize; ///< ファイルを読み取るためのバッファサイズ
  unsigned int width;               ///< AVIファイルの幅
  unsigned int height;              ///< AVIファイルの高さ
  unsigned int reserved[4];         ///< 予約領域(0)
};

struct RECT
{
  int left;   ///< 左
  int top;    ///< 上
  int right;  ///< 右
  int bottom; ///< 下
};



struct AVIStreamHeader
{
  unsigned int  type;                 ///< ストリームに含まれるデータのタイプ
  unsigned int  handler;              ///< 特定のデータ ハンドラを示す(いわゆるコーデック)
  unsigned int  flags;                ///< データ ストリームに対するフラグ
                                      ///<   0x00000001 - AVISF_DISABLED
                                      ///<   0x00010000 - AVISF_VIDEO_PALCHANGES
  unsigned int  priority;             ///< ストリームタイプの優先順位
  unsigned int  initialFrames;        ///< AVIシーケンスの開始フレームより前にあるフレーム数
  unsigned int  scale;                ///< このストリームが使用するタイム スケール
  unsigned int  rate;                 ///< サンプリングレート(1 秒あたりのサンプル数 = rate/scale)
  unsigned int  start;                ///< AVIファイルの開始時間
  unsigned int  length;               ///< このストリームの長さ
  unsigned int  suggestedBufferSize;  ///< このストリームを読み取るために必要なバッファの大きさ
  unsigned int  quality;              ///< ストリーム内のデータの品質
  unsigned int  sampleSize;           ///< 1 サンプルのサイズ
  RECT          frame;                ///< テキストまたはビデオ ストリームに対する転送先矩形
};

struct AVIINDEXENTRY
{
  unsigned int ckid;        ///< チャンク識別子('00db', '00wb'等）
  unsigned int flags;       ///< フラグ
                            ///<   0x00000001L - AVIIF_LIST (chunk is a 'LIST')
                            ///<   0x00000010L - AVIIF_KEYFRAME (this frame is a key frame)
                            ///<   0x00000100L - AVIIF_NOTIME (this frame doesn't take any time)
                            ///<   0x0FFF0000L - AVIIF_COMPUSE (these bits are for compressor use)
  unsigned int chunkOffset; ///< チャンクの位置
  unsigned int chunkLength; ///< チャンクのサイズ
};
