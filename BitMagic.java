
/**
 * cf. http://d.hatena.ne.jp/siokoshou/20090704
 *     http://d.hatena.ne.jp/siokoshou/20090706
 */
final class BitMagic {
  private static final long LOG2_HASH = 0x03f566ed27179461L; // マジックナンバー
  private static final int[] log2_hash_table;
  
  static {
    log2_hash_table = new int[64];
    for (int i = 0; i < 64; ++i) {
      //log2_hash_table[(int)((LOG2_HASH*(1L<<i))>>>58)] = i;
      log2_hash_table[(int)((LOG2_HASH<<i)>>>58)] = i; // 上と同義
    }
  }

  private static boolean check(long x) {
    int bitflg = 0;

    for (int i = 0; i < 64; ++i) {
      int n = (int)((x<<i)>>>58);
      bitflg |= (1L<<n);
    }
    
    return ~bitflg == 0;
  }

  public static void main(String[] args) {
    //全てのマジックナンバーをM系列を使って計算
    {
      java.util.HashSet<String> magic_number_set = new java.util.HashSet<String>();
      int p = 6;
      for (int x0 = 0; x0 < (1<<p); ++x0) { // 全種類の初期値(=0..2^p-1)について
        for (int q = 1; q < p; ++q) { // 全種類のq(=1..q-1)について
          long x = x0; // 初期値を設定
          for (int i = 0; i < (1<<p)-p; ++i) { // 残りの系列を生成
            x <<= 1;
            x  |= ((x>>>p)&1) ^ ((x>>>q)&1);
          }
          if (check(x)) { // p,qが既約かつ原始かチェック
            System.err.printf("(%d, %d; %2d)\n", p, q, x0);
            for (int i = 0; i < 64; ++i) {
              String hex = Long.toHexString(x);
              while (hex.length() < 16) {
                hex = "0" + hex;
              }
              hex = "0x" + hex;
              magic_number_set.add(hex);
              System.err.println(hex);
              x = (x<<1) | (x>>>(64-1)); // 回転
            }
          }
        }
      }
      String[] numbers = magic_number_set.toArray(new String[magic_number_set.size()]);
      java.util.Arrays.sort(numbers);
      for (String str : numbers) {
        System.out.println(str);
      }
    }

    // テクスチャのミップマップレベルを計算する
    int x = 123451234;
    x = reverse(x);
    x = x & -x; // 最も右に立っているビットだけを残す
    x = reverse(x);
    int n = ntz(x);
  }

  // 立ってるビット数を数える
  private static int count(int x) {
    x = ((x&0xAAAAAAAA)>>> 1) + (x&0x55555555);
    x = ((x&0xCCCCCCCC)>>> 2) + (x&0x33333333);
    x = ((x&0xF0F0F0F0)>>> 4) + (x&0x0F0F0F0F);
    x = ((x&0xFF00FF00)>>> 8) + (x&0x00FF00FF);
    x = ((x&0xFFFF0000)>>>16) + (x&0x0000FFFF);
    return x;
  }

  private static long count(long x) {
    x = ((x&0xAAAAAAAAAAAAAAAAL)>>> 1) + (x&0x5555555555555555L);
    x = ((x&0xCCCCCCCCCCCCCCCCL)>>> 2) + (x&0x3333333333333333L);
    x = ((x&0xF0F0F0F0F0F0F0F0L)>>> 4) + (x&0x0F0F0F0F0F0F0F0FL);
    x = ((x&0xFF00FF00FF00FF00L)>>> 8) + (x&0x00FF00FF00FF00FFL);
    x = ((x&0xFFFF0000FFFF0000L)>>>16) + (x&0x0000FFFF0000FFFFL);
    x = ((x&0xFFFFFFFF00000000L)>>>32) + (x&0x00000000FFFFFFFFL);
    return x;
  }

  // ビットを逆順にする
  private static int reverse(int x) {
    x = ((x&0xAAAAAAAA)>>> 1) | ((x&0x55555555)<< 1);
    x = ((x&0xCCCCCCCC)>>> 2) | ((x&0x33333333)<< 2);
    x = ((x&0xF0F0F0F0)>>> 4) | ((x&0x0F0F0F0F)<< 4);
    x = ((x&0xFF00FF00)>>> 8) | ((x&0x00FF00FF)<< 8);
    x = ((x&0xFFFF0000)>>>16) | ((x&0x0000FFFF)<<16);
    return x;
  }

  private static long reverse(long x) {
    x = ((x&0xAAAAAAAAAAAAAAAAL)>>> 1) | ((x&0x5555555555555555L)<< 1);
    x = ((x&0xCCCCCCCCCCCCCCCCL)>>> 2) | ((x&0x3333333333333333L)<< 2);
    x = ((x&0xF0F0F0F0F0F0F0F0L)>>> 4) | ((x&0x0F0F0F0F0F0F0F0FL)<< 4);
    x = ((x&0xFF00FF00FF00FF00L)>>> 8) | ((x&0x00FF00FF00FF00FFL)<< 8);
    x = ((x&0xFFFF0000FFFF0000L)>>>16) | ((x&0x0000FFFF0000FFFFL)<<16);
    x = ((x&0xFFFFFFFF00000000L)>>>32) | ((x&0x00000000FFFFFFFFL)<<32);
    return x;
  }

  // x=2^nとなる整数xのnを求める
  // cf. http://d.hatena.ne.jp/siokoshou/20090704#p1
  private static int ntz(long x) {
    if (x==0) return 64;
    return log2_hash_table[(int)((LOG2_HASH*x)>>>58)];
  }

  private static void printBits(int x) {
    for (int i=32-1; i>=0; --i) {
      System.out.print((((x>>i)&1)==0)?'0':'1');
    }
    System.out.println();
  }

  private static void printBits(long x) {
    for (int i=64-1; i>=0; --i) {
      System.out.print((((x>>i)&1)==0)?'0':'1');
    }
    System.out.println();
  }

  private static void printReverseBits(int x) {
    for (int i=0; i<32; ++i) {
      System.out.print((((x>>i)&1)==0)?'0':'1');
    }
    System.out.println();
  }

  private static void printReverseBits(long x) {
    for (int i=0; i<64; ++i) {
      System.out.print((((x>>i)&1)==0)?'0':'1');
    }
    System.out.println();
  }
}
