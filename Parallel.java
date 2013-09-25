import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Callable;

// cf. http://stackoverflow.com/questions/4010185/parallel-for-for-java
public final class Parallel {
  private static final int nCPUs = Runtime.getRuntime().availableProcessors();

  public static interface Body<T> {
    public void run(final T t);
  }

  public static <T> void For(final Iterable<T> params, final Body<T> body) {
    final List<Callable<Void>> tasks = new ArrayList<>();
    for (final T param : params) {
      tasks.add(new Callable<Void>() {
        @Override
        public Void call() {
          body.run(param);
          return null;
        }
      });
    }

    final ExecutorService executor = Executors.newFixedThreadPool(nCPUs);
    try {
      executor.invokeAll(tasks);
    } catch (final InterruptedException e) {
    } finally {
      executor.shutdown();
    }
  }

  // test
  public static void main(final String[] args) {
    List<Integer> messages = new ArrayList<>();
    for (int i = 0; i < 10; ++i) {
      messages.add(i);
    }

    Parallel.For(messages, new Parallel.Body<Integer>() {
      @Override
      public void run(final Integer value) {
        System.out.println(value);
      }
    });
  }
}

