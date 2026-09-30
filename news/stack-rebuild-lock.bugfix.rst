Fix a race that could corrupt the capture file when a thread that was already running when tracking started made its first Python call while other threads were allocating memory.
