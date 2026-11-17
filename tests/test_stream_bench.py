import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stream_bench import SSEDecoder


class SSEDecoderTest(unittest.TestCase):
    def test_split_utf8_bom_crlf_and_multiline(self):
        data=b"\xef\xbb\xbf: comment\r\ndata: "+"你好".encode()+b"\r\ndata: second\r\n\r\ndata: [DONE]\n\n"
        decoder=SSEDecoder()
        events=[]
        for byte in data:
            events.extend(decoder.feed(bytes([byte])))
        self.assertEqual(events,["你好\nsecond","[DONE]"])

    def test_unterminated_event_is_not_completion(self):
        self.assertEqual(list(SSEDecoder().feed(b"data: [DONE]\n")),[])

    def test_data_bounds(self):
        with self.assertRaises(ValueError):
            list(SSEDecoder(limit=16).feed(b"data: 1234567890\ndata: 1234567890\n\n"))


if __name__=="__main__":
    unittest.main()
