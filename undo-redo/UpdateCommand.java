public class UpdateCommand implements ICommand {
  private DataModel model_;
  private String prevText_;
  private String nextText_;
  
  public UpdateCommand( DataModel model, String newText ) {
    model_ = model;
    nextText_ = newText;
  }
  
  public void invoke() {
    prevText_ = model_.getText();
    model_.setText( nextText );
  }
  
  public void undo() {
    model_.setText( prevText_ );
  }
  
  public void redo() {
    model_.setText( nextText_ );
  }
}
