#include <iostream>

int main()
{
  typedef unsigned long long int ulong_t;

  // �r�b�g�������Ă��镔���W����񋓂���
  ulong_t const d = 0x7;

  ulong_t n = 0;
  do {
    std::cout << n << std::endl;
    n = (n - d) & d;
  } while (n);

  return 0;
}
