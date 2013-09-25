public class CommandInvoker {
  private Stack<ICommand> undoStack_;
  private Stack<ICommand> redoStack_;
  
  public CommandInvoker() {
    undoStack_ = Stack<ICommand>();
    redoStack_ = Stack<ICommand>();
  }
  
  public void invoke( ICommand command ) {
    command.invoke();
    redoStack_.clear();
    undoStack_.push( command );
  }
  
  public void undo() {
    if ( undoStack_.isEmpty() ) {
      return;
    }
    ICommand command = undoStack_.pop();
    command.undo();
    redoStack_.push( command );
  }
  
  public void redo() {
    if ( redoStack_.isEmpty() ) {
      return;
    }
    ICommand command = redoStack_.pop();
    command.redo();
    undoStack_.push( command );
  }
}
