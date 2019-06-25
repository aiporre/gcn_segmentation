from config import DEBUG
import traceback
def print_debug(message, exception=None):
    if DEBUG and not exception:
        print(message)
    elif DEBUG and exception:
        print(exception)
        traceback.print_exc()

