"""The policy screen must distinguish C++ code from comments and literals."""
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_cpp_style import code_only, screen, source_files


class CodeScreenTests(unittest.TestCase):
    def test_comments_and_literals_are_not_code(self):
        source = '// throw Bad();\n/* catch (...) {} */\nauto s = "std::filesystem";\nthrow Bad();\n'
        clean = code_only(source)
        self.assertEqual(re.findall(r'\bthrow\b', clean), ['throw'])
        self.assertNotIn('catch', clean)
        self.assertNotIn('filesystem', clean)
        self.assertEqual(source.count('\n'), clean.count('\n'))

    def test_raw_string_can_contain_quotes_and_keywords(self):
        source = 'auto s = R"tag(throw "x";\ncatch (...) {})tag";\nthrow Error();'
        clean = code_only(source)
        self.assertEqual(re.findall(r'\bthrow\b', clean), ['throw'])
        self.assertNotIn('catch', clean)
        self.assertEqual(source.count('\n'), clean.count('\n'))

    def test_numeric_separators_do_not_swallow_following_code(self):
        source = "auto x = 0xA'B'C; auto y = 1'000; throw Error();"
        self.assertIn('throw', code_only(source))

    def test_prefixed_character_and_escaped_string(self):
        source = '''auto x = L'a'; auto s = "quote: \\" catch"; throw Error();'''
        self.assertIn('throw', code_only(source))
        self.assertNotIn('catch', code_only(source))


class PolicyScreenTests(unittest.TestCase):
    def test_disallowed_headers_and_ratio_usage_are_reported(self):
        source = (
            '// #include <ratio>\n'
            'const char* example = "#include <cfenv>";\n'
            '#include <ratio>\n'
            '#include /* documented */ "fenv.h"\n'
            'using Example = std::ratio<1, 2>;\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example.cc").write_text(source)
            with patch("scripts.check_cpp_style.ROOT", root):
                findings = screen([Path("example.cc")])
        self.assertEqual(
            sorted((item["line"], item["rule"]) for item in findings),
            [(3, "disallowed_standard_library"), (4, "disallowed_standard_library"),
             (5, "disallowed_standard_library")],
        )


class SourceFilesTests(unittest.TestCase):
    def test_working_tree_rename_includes_the_new_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            header = root / "fragment.hpp"
            header.write_text("void Function() {}\n")
            subprocess.run(["git", "add", header.name], cwd=root, check=True)
            fragment = header.with_suffix(".inc")
            header.rename(fragment)

            with patch("scripts.check_cpp_style.ROOT", root):
                self.assertEqual(source_files(), [Path(fragment.name)])


if __name__ == '__main__':
    unittest.main()
