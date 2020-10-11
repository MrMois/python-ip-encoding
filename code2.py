
def byte_to_bitarr(byte):
    assert(byte <= 255 and byte >= 0)

    byte = np.uint8(byte)
    bits = [(byte & (1 << 7-b)) != 0 for b in range(8)]

    return np.array(bits, dtype=np.uint8)


def encode(bytes):
    assert(len(bytes) == 3)

    bytes = [byte_to_bitarr(b) for b in bytes]

    pass
