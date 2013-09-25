/**
 * 修正モートン順序則
 *   座標を二進数で表現し，先頭から交互に並べる．
 *  それを十進数で書き直し，小さい順に並べる．
 */

/// x = XxXx XxXx XxXx XxXx XxXx XxXx XxXx XxXx (16)
/// return xxxx xxxx xxxx xxxx (16)
inline std::size_t compact2(std::size_t x)
{
                                                    // 55 55 55 55
  x = (x & 0x11111111U) | ((x >> 1) & 0x22222222U); // 33 33 33 33
  x = (x & 0x03030303U) | ((x >> 2) & 0x0C0C0C0CU); // 0F 0F 0F 0F
  x = (x & 0x000F000FU) | ((x >> 4) & 0x00F000F0U); // 00 FF 00 FF
  x = (x & 0x000000FFU) | ((x >> 8) & 0x0000FF00U); // 00 00 FF FF
  return x;
}

/// x = xxxx xxxx xxxx xxxx (16)
/// return 0x0x 0x0x 0x0x 0x0x 0x0x 0x0x 0x0x 0x0x (16)
inline std::size_t expand2(std::size_t x)
{
                                                    // 00 00 FF FF
  x = (x & 0x000000FFU) | ((x & 0x0000FF00U) << 8); // 00 FF 00 FF
  x = (x & 0x000F000FU) | ((x & 0x00F000F0U) << 4); // 0F 0F 0F 0F
  x = (x & 0x03030303U) | ((x & 0x0C0C0C0CU) << 2); // 33 33 33 33
  x = (x & 0x11111111U) | ((x & 0x22222222U) << 1); // 55 55 55 55
  return x;
}

/// x = xxxx xxxx xxxx xxxx (16)
/// y = XXXX XXXX XXXX XXXX (16)
/// return XxXx XxXx XxXx XxXx XxXx XxXx XxXx XxXx (16)
inline std::size_t marge2(std::size_t x, std::size_t y)
{
  return expand2(x) | (expand2(y) << 1);
}

/// x = XXx XXx XXx XXx XXx XXx XXx XXx XXx XXx (8)
/// return 0000 00xx xxxx xxxx
inline std::size_t compact3(std::size_t x)
{
                                                                                // 1 111 111 111
  x = (x & 01111111111U) | ((x>> 2) & 02222222222U) | ((x>>4) & 04444444444U);  // 7 007 007 007
  x = (x & 00007000007U) | ((x>> 6) & 00070000070U);                            // 0 077 000 077
  x = (x & 00000000077U) | ((x>>12) & 00000007700U);                            // 0 000 007 777
  return x;
}

/// x = 00x xxx xxx xxx
/// return 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x (8)
inline std::size_t expand3(std::size_t x)
{
                                                                                  // 0 000 007 777
  x = (x & 00000000077U) | ((x & 00000007700U) <<12);                             // 0 077 000 077
  x = (x & 00007000007U) | ((x & 00070000070U) << 6);                             // 7 007 007 007
  x = (x & 01001001001U) | ((x & 02222222222U) << 2) | ((x & 04444444444U) << 4); // 1 111 111 111
  return x;
}

/// x = xxxx xxxx xxxx xxxx
/// y = XXXX XXXX XXXX XXXX
/// return XxXx XxXx XxXx XxXx XxXx XxXx XxXx XxXx
inline
std::size_t
  marge3(
    std::size_t const x,
    std::size_t const y,
    std::size_t const z)
{
  return expand3(x) | (expand3(y) << 1) | (expand3(z) << 2);
}

/// 修正モートン順序則
int _tmain(int argc, _TCHAR* argv[])
{
#if 0
  // i -> (x, y) -> j
  for (std::size_t i = 0; i < (1<<15)*(1<<15); ++i) {
    std::size_t const x = compact2(i   );
    std::size_t const y = compact2(i>>1);
    std::size_t const j = marge2(x, y);
    if (i != j) {
      std::fprintf(stdout, "%2Iu -> (x=%2Iu, y=%2Iu) -> %2Iu\n", i, x, y, j);
    }
  }
#else
  // i -> (x, y) -> j
  for (std::size_t i = 0; i < (1<<9)*(1<<9)*(1<<9); ++i) {
    std::size_t const x = compact3(i   );
    std::size_t const y = compact3(i>>1);
    std::size_t const z = compact3(i>>2);
    std::size_t const j = marge3(x, y, z);
    if (i != j) {
      std::fprintf(stdout, "%2Iu -> (x=%2Iu, y=%2Iu, z=%2Iu) -> %2Iu\n", i, x, y, z, j);
    }
  }
#endif
  std::fputs("ok", stdout);
  std::getchar();

  return 0;
}
