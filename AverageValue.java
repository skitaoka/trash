/**
 * 平均値の計算
 * avg_{n} = (data_1 + data_2 + ... + data_n) / n;
 *
 * これまでのデータ数と平均値を保持しておけば計算できる。
 * avg_{n} = ((n - 1) * avg_{n-1} + data_n) / n;
 */
public final class AverageValue {
  private double average_;
  private int count_;

  public void clear() {
    average_ = 0;
    count_ = 0;
  }

  public void addValue(final double value) {
    average_ = (count_ * average_ + value) / (count_ + 1);
    count_++;
  }

  public double getAverage() {
    return average_;
  }

  public int getCount() {
    return count_;
  }
}
