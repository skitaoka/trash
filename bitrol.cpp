// ビットを回転させる。
// T must be unsigned integer.
template <typename T>
unsigned ROL(T const dt, int const bits)
{
  return (dt << n) | (dt >> (sizeof(T) * 8 - bits));
}
