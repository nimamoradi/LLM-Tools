from enum import Enum

class UserInput(Enum):
    MESSAGE = "message"

class SystemOutput(Enum):
    STREAM = "stream"
    TEXT = "text"
    EXIT = "exit"
