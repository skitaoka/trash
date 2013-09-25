/*
 * A*アルゴリズム
 */

/**
 * 「状態」を表すデータ構造 State は，
 * 現状から移行し得るすべての状態を列挙するメソッド
 * enum_trans と，同一演算子 "==" を持つものとする。
 * ファンクタ GFunc と HFunc はそれぞれ関数 G, H に相当し，
 * State::Value 型の値を返すものとする。
 */
template <typename State, typename Stack, typename GFunc, typename HFunc>
void Astar( const State& start, Stack& result )
{
}
