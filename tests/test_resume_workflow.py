"""Verify artifact tracing without invoking remote services or embedding models."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.resume_parser import ResumeError
from scripts.run_resume import RecordedEmbeddings, RunLog, main, write_manifest


class ResumeWorkflowTests(unittest.TestCase):
    def test_record_exact_vector_used_for_query(self):
        with tempfile.TemporaryDirectory() as folder:
            vector = [1.0] + [0.0] * 1023
            delegate = Mock()
            delegate.embed_query.return_value = vector
            recorder = RecordedEmbeddings(delegate, Path(folder), {'model': 'test', 'revision': 'fixed'})
            result = recorder.embed_query('Python developer')
            self.assertIs(result, vector)
            delegate.embed_query.assert_called_once_with('Python developer')
            payload = json.loads((Path(folder) / 'query_embedding.json').read_text())
            self.assertEqual(payload['vector'], vector)
            self.assertEqual(payload['dimension'], 1024)
            self.assertEqual(payload['l2_norm'], 1)
            self.assertEqual(payload['query_sha256'], hashlib.sha256(b'Python developer').hexdigest())
            self.assertTrue(recorder.recorded)

    def test_stage_failure_log_omits_exception_content(self):
        with tempfile.TemporaryDirectory() as folder:
            log = RunLog(Path(folder))
            with self.assertRaises(ValueError):
                with log.stage('pdf_extraction'):
                    raise ValueError('private resume text')
            text = (Path(folder) / 'events.jsonl').read_text()
            self.assertNotIn('private resume text', text)
            events = [json.loads(line) for line in text.splitlines()]
            self.assertEqual([event['status'] for event in events], ['started', 'failed'])
            self.assertGreaterEqual(events[-1]['duration_seconds'], 0)

    def test_manifest_covers_exact_file_content(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'response.json').write_text('{}')
            write_manifest(output, status='completed')
            write_manifest(output, status='completed')
            manifest = json.loads((output / 'run_manifest.json').read_text())
            self.assertEqual(manifest['files'], [{'path': 'response.json', 'bytes': 2,
                                                  'sha256': hashlib.sha256(b'{}').hexdigest()}])

    def test_cli_preserves_failed_manifest_and_error_code(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Mock(artifacts_dir=Path(folder))
            settings.model_copy.return_value = settings
            with patch('scripts.run_resume.get_settings', return_value=settings), patch(
                    'scripts.run_resume.run', side_effect=ResumeError('not_pdf', 'Resume must be a PDF.')):
                self.assertEqual(main(['invalid.pdf']), 1)
            runs = list((Path(folder) / 'resume_runs').iterdir())
            self.assertEqual(len(runs), 1)
            manifest = json.loads((runs[0] / 'run_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['error_code'], 'not_pdf')
            self.assertTrue((runs[0] / 'events.jsonl').exists())
            self.assertFalse((runs[0] / 'response.json').exists())


if __name__ == '__main__':
    unittest.main()
