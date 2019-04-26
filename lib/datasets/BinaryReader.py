import struct

class BinaryReaderEOFException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'Not enough bytes in file to satisfy read request'

class BinaryReader(object):

    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.typeNames = {
            'int8': 'b',
            'uint8': 'B',
            'int16': 'h',
            'uint16': 'H',
            'int32': 'i',
            'uint32': 'I',
            'int64': 'q',
            'uint64': 'Q',
            'float': 'f',
            'double': 'd',
            'char': 's'}

    def read(self, typeName, times=1):
        typeFormat = self.typeNames[typeName.lower()]*times
        typeSize = struct.calcsize(typeFormat)
        value = self.file.read(typeSize)
        if typeSize != len(value):
            raise BinaryReaderEOFException
        return struct.unpack(typeFormat, value)

    def close(self):
        self.file.close()

