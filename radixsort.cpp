//
// 基数ソートの段階的な実装.
//   基数ソートは安定なソートなのでいろいろと便利.
//

#include <memory>
#include <iterator>
#include <iostream>

typedef unsigned char uint8;
typedef unsigned int uint32;

// 重複のない 0-255 の範囲の整数値のソート.
void sort1()
{
  uint8 const values[] = { 54, 18, 2, 128, 3 };
  std::size_t const length = sizeof(values) / sizeof(values[0]);

  int bucket[256];
  std::memset(bucket, -1, 256 * sizeof(int)); // Fill with -1

  for (std::size_t i = 0; i < length; ++i) {
    uint8 const c = values[i];
    bucket[c] = c;
  }

  for (std::size_t i = 0; i < 256; ++i) {
    if (bucket[i] != -1) {
      std::cout << bucket[i] << std::endl;
    }
  }
}

// 重複のある 0-255 の範囲の整数値のソート.
void sort2()
{
  uint8 const values[] = { 54, 18, 2, 128, 3, 128, 3, 4 };
  std::size_t const length = sizeof( values ) / sizeof( values[0] );

  std::size_t counters[256];
  std::memset(counters, 0, 256 * sizeof(std::size_t)); // Set all counters to 0

  for (std::size_t i = 0; i < length; ++i) {
    uint8 const c = values[i];
    counters[c]++;
  }

  for (std::size_t i = 0; i < 256; ++i) {
    for (std::size_t j = 0; j < counters[i]; ++j) {
      std::cout << i << std::endl;
    }
  }
}

// 重複のある 0-255 の範囲の整数値のソート.
// ヒストグラムを作ってデータを書き出す.
void sort3()
{
  uint8 const values[] = { 54, 18, 2, 128, 3, 128, 3, 4 };
  std::size_t const length = sizeof(values) / sizeof(values[0]);
  uint8 dest[length];

  std::size_t counters[256];
  std::memset(counters, 0, 256 * sizeof(std::size_t)); // Set all counters to 0

  for (std::size_t i = 0; i < length; ++i) {
    uint8 c = values[i];
    counters[c]++;
  }

  std::size_t offset[256];
  offset[0] = 0;
  for (std::size_t i = 1; i < 256; ++i) {
    offset[i] = offset[i-1] + counters[i-1];
  }

  for (std::size_t i = 0; i < length; ++i) {
    uint8 const c = values[i];
    dest[offset[c]++] = c;
  }

  std::copy(dest, dest + length, std::ostream_iterator<int>(std::cout, "\n"));
}

// 重複のある非負数値のソート.
// 4 パスで全体をソートする.
void sort4()
{
  uint32 values[] = { 2083, 660, 2785, 4906, 8080, 10050, 10116, 16343, 10578, 12974 };
  std::size_t const length = sizeof(values) / sizeof(values[0]);
  uint32 dest[length];

  std::size_t counters[256];
  std::size_t offset  [256];

  uint32 * input  = values;
  uint32 * output = dest;

  for (std::size_t pass = 0; pass < 4; ++pass) {
    std::memset(counters, 0, 256 * sizeof(std::size_t)); // Set all counters to 0

    for (std::size_t i = 0; i < length; ++i) {
      uint8 const radix = (input[i] >> (pass << 3)) & 0xFF;
      counters[radix]++;
    }

    offset[0] = 0;
    for (std::size_t i = 1; i < 256; ++i) {
      offset[i] = offset[i-1] + counters[i-1];
    }

    for (std::size_t i = 0; i < length; ++i) {
      uint8 const radix = (input[i] >> (pass << 3)) & 0xFF;
      output[offset[radix]++] = input[i];
    }

    std::swap(input, output);
  }

  std::copy(values, values + length, std::ostream_iterator<int>(std::cout, "\n"));
}

// 非負 float 値のソート.
// IEEE 浮動小数点フォーマットは
//   x > y > 0 => IR(x) > IR(y)
// となるので、非負値ならそのままソートできる
void sort5()
{
  float values[] = {2083, 660, 2785, 4906, 8080, 10050, 10116, 16343, 10578, 12974};
  std::size_t const length = sizeof(values) / sizeof(values[0]);
  float dest[length];

  std::size_t counters[256];
  std::size_t offset  [256];

  uint32 * input  = reinterpret_cast<uint32 *>(values);
  uint32 * output = reinterpret_cast<uint32 *>(dest);

  for (std::size_t pass = 0; pass < 4; ++pass) {
    std::memset(counters, 0, 256 * sizeof(std::size_t));  // Set all counters to 0

    for (std::size_t i = 0; i < length; ++i) {
      uint8 const radix = (input[i] >> (pass << 3)) & 0xFF;
      counters[radix]++;
    }

    offset[0] = 0;
    for (std::size_t i = 1; i < 256; ++i) {
      offset[i] = offset[i-1] + counters[i-1];
    }

    for (std::size_t i = 0; i < length; ++i) {
      uint8 const radix = (input[i] >> (pass << 3)) & 0xFF;
      output[offset[radix]++] = input[i];
    }

    std::swap(input, output);
  }

  std::copy(values, values + length, std::ostream_iterator<float>(std::cout, "\n"));
}

// float 値のソート.
// x < y < 0 => IR(x) > IR(y)
// 負数の一番小さいものが一番大きいものと判定される
void sort6()
{
  float values[] = { 2083, -660, 2785, -4906, 8080, -10050, 10116, -16343, 10578, 12974 };
  std::size_t const length = sizeof(values) / sizeof(values[0]);
  float dest[length];

  std::size_t counters[256];
  std::size_t offset  [256];

  uint32 * input  = reinterpret_cast<uint32 *>(values);
  uint32 * output = reinterpret_cast<uint32 *>(dest);

  for (std::size_t pass = 0; pass < 3; ++pass) {
    std::memset(counters, 0, 256 * sizeof(std::size_t)); // Set all counters to 0

    for (std::size_t i = 0; i < length; ++i) {
      uint8 const radix = (input[i] >> (pass << 3)) & 0xFF;
      counters[radix]++;
    }

    offset[0] = 0;
    for (std::size_t i = 1; i < 256; ++i) {
      offset[i] = offset[i-1] + counters[i-1];
    }

    for (std::size_t i = 0; i < length; ++i) {
      uint8 const radix = ( input[i] >> ( pass << 3 ) ) & 0xFF;
      output[offset[radix]++] = input[i];
    }

    std::swap(input, output);
  }
  {
    std::memset(counters, 0, 256 * sizeof(std::size_t)); // Set all counters to 0

    for (std::size_t i = 0; i < length; ++i) {
      uint8 const radix = (input[i] >> 24) & 0xFF;
      counters[radix]++;
    }

    // 負数の数を数える
    std::size_t negative = 0;
    for (std::size_t i = 128; i < 256; ++i) {
      negative += counters[i];
    }

    // 正数のオフセットを作る
    offset[0] = negative;
    for (std::size_t i = 1; i < 128; ++i) {
      offset[i] = offset[i-1] + counters[i-1];
    }

    // 負数のオフセットを逆から作る
    offset[255] = 0;
    for (std::size_t i = 0; i < 127; ++i) {
      offset[254-i] = offset[255-i] + counters[255-i];
    }

    // 安定ソートの特性から負数の場合は逆順から読まないとダメなので
    // そのための処置
    for (std::size_t i = 128; i < 256; ++i) {
      offset[i] += counters[i];
    }

    for (std::size_t i = 0; i < length; ++i) {
      uint8 radix = (input[i] >> 24) & 0xFF;
      if (radix & 0x80) {
        output[ --offset[radix] ] = input[i]; // 負数は逆から読む
      } else {
        output[ offset[radix]++ ] = input[i];
      }
    }

    std::swap(input, output);
  }

  std::copy(values, values + length, std::ostream_iterator<float>(std::cout, "\n"));
}

// float 値のソート.
// 0xFFFFFFFF のビットパターンで表現される nan 以外のすべての float 値を正しくソートできる.
void sort7()
{
  float values[] = { 10578, 2083, -660, 2785, -4906, 8080, -10050, 10116, 12974, -16343 };
  std::size_t const length = sizeof(values) / sizeof(values[0]);
  float dest[length];

  std::size_t counters[256];
  std::size_t offset  [256];

  uint32 * input  = reinterpret_cast<uint32 *>(values);
  uint32 * output = reinterpret_cast<uint32 *>(dest);

  for (std::size_t pass = 0; pass < 4; ++pass) {
    std::memset(counters, 0, 256 * sizeof(std::size_t));  // Set all counters to 0

    for (std::size_t i = 0; i < length; ++i) {
      // 負数なら 0xFFFFFFFF, 非負数なら 0x80000000 となるマスクを作る.
      // マスクと xor をとることで IEEE float は整数として比較できるビット列になる.
      uint32 const mask  = -int(input[i] >> 31) | 0x80000000;
      uint8  const radix = ((input[i] ^ mask) >> (pass << 3)) & 0xFF;
      counters[radix]++;
    }

    offset[0] = 0;
    for (std::size_t i = 1; i < 256; ++i) {
      offset[i] = offset[i-1] + counters[i-1];
    }

    for (std::size_t i = 0; i < length; ++i) {
      // 負数なら 0xFFFFFFFF, 非負数なら 0x80000000 となるマスクを作る.
      // マスクと xor をとることで IEEE float は整数として比較できるビット列になる.
      uint32 const mask  = -int(input[i] >> 31) | 0x80000000;
      uint8  const radix = ((input[i] ^ mask) >> (pass << 3)) & 0xFF;
      output[offset[radix]++] = input[i];
    }

    std::swap(input, output);
  }

  std::copy(values, values + length, std::ostream_iterator<float>(std::cout, "\n"));
}


int main( void )
{
  std::cout << std::endl << "sort1()" << std::endl;
  sort1();

  std::cout << std::endl << "sort2()" << std::endl;
  sort2();

  std::cout << std::endl << "sort3()" << std::endl;
  sort3();

  std::cout << std::endl << "sort4()" << std::endl;
  sort4();

  std::cout << std::endl << "sort5()" << std::endl;
  sort5();

  std::cout << std::endl << "sort6()" << std::endl;
  sort6();

  std::cout << std::endl << "sort7()" << std::endl;
  sort7();

  return 0;
}
