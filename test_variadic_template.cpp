//
// 可変長 template による tuple の実装テスト
//
#include <iostream>

template <int N, typename ...Ts>
struct my_tuple_data;
    
template <int N>
struct my_tuple_data<N>
{
};

template <int N, typename Head, typename ...Body>
struct my_tuple_data<N, Head, Body...>: public my_tuple_data<N+1, Body...>
{
    Head value;
};

template <typename ...Ts>
struct my_tuple: public my_tuple_data<0, Ts...>
{
};

template <int N, typename Head, typename ...Body>
struct my_tuple_type
{
    typedef typename my_tuple_type<N-1, Body...>::value_type value_type;
    typedef typename my_tuple_type<N-1, Body...>::tuple_type tuple_type;
};

template <typename Head, typename ...Body>
struct my_tuple_type<0, Head, Body...>
{
    typedef Head                            value_type;
    typedef my_tuple_data<0, Head, Body...> tuple_type;
};


template <int N, typename ...Ts> inline
typename my_tuple_type<N, Ts...>::value_type &
    get(my_tuple<Ts...>& t)
{
  typedef typename my_tuple_type<N, Ts...>::tuple_type & tuple_type;
  return ((tuple_type&) t).value;
}


int main()
{
  std::ios::sync_with_stdio(false);

  my_tuple<int, double, char const *> data;

  get<0>(data) = 10;
  get<1>(data) = 1.0;
  get<2>(data) = "Hello world!";
  
  std::cout << get<0>(data) << std::endl;
  std::cout << get<1>(data) << std::endl;
  std::cout << get<2>(data) << std::endl;
  
  return 0;
}
