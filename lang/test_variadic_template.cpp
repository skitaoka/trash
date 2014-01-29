//
// 可変長 template による tuple の実装テスト
//
#include <iostream>

template <typename ...Ts>
struct my_tuple_data;
    
template <>
struct my_tuple_data<>
{
};

template <typename Head, typename ...Body>
struct my_tuple_data<Head, Body...>: public my_tuple_data<Body...>
{
    Head value;
};

template <typename ...Ts>
struct my_tuple: public my_tuple_data<Ts...>
{
};

template <int I, typename Head, typename ...Body>
struct my_tuple_type
{
    typedef typename my_tuple_type<I-1, Body...>::value_type value_type;
    typedef typename my_tuple_type<I-1, Body...>::tuple_type tuple_type;
};

template <typename Head, typename ...Body>
struct my_tuple_type<0, Head, Body...>
{
    typedef Head                         value_type;
    typedef my_tuple_data<Head, Body...> tuple_type;
};


template <int N, typename ...Ts>
typename my_tuple_type<N, Ts...>::value_type &
    get(my_tuple<Ts...>& t)
{
  typedef typename my_tuple_type<N, Ts...>::tuple_type tuple_type;
  return static_cast<tuple_type&>(t).value;
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
