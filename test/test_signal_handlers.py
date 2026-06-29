import signal
import unittest

import hypatorch


class TestSignalHandlerChaining(unittest.TestCase):
    """The training signal handler must request a graceful stop AND chain to a
    previously-installed handler, so an outer layer (e.g. a tracking-run
    finalizer) can react immediately instead of waiting for the next
    should_stop check.
    """

    def _invoke_installed(self, signum):
        handler = signal.getsignal(signum)
        handler(signum, None)

    def test_chains_to_previous_custom_handler_and_requests_stop(self):
        trainer = hypatorch.Trainer()
        trainer.should_stop = False
        called = {}

        def previous(signum, frame):  # noqa: ANN001
            called["signum"] = signum

        prev = signal.signal(signal.SIGTERM, previous)
        try:
            with trainer._signal_handler_context():
                self._invoke_installed(signal.SIGTERM)
        finally:
            signal.signal(signal.SIGTERM, prev)

        self.assertTrue(trainer.should_stop)
        self.assertEqual(trainer._stop_reason, "signal:SIGTERM")
        self.assertEqual(called.get("signum"), signal.SIGTERM)

    def test_previous_handler_that_raises_propagates(self):
        trainer = hypatorch.Trainer()

        class _Abort(Exception):
            pass

        def previous(signum, frame):  # noqa: ANN001
            raise _Abort()

        prev = signal.signal(signal.SIGTERM, previous)
        try:
            with self.assertRaises(_Abort):
                with trainer._signal_handler_context():
                    self._invoke_installed(signal.SIGTERM)
        finally:
            signal.signal(signal.SIGTERM, prev)

        # The graceful stop was still requested before propagating.
        self.assertTrue(trainer.should_stop)

    def test_default_handler_is_not_chained(self):
        # With SIG_DFL installed (nothing custom), the handler must still request
        # stop without trying to invoke the non-callable default (which is also
        # how the graceful-only behavior is preserved).
        trainer = hypatorch.Trainer()
        trainer.should_stop = False
        prev = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            with trainer._signal_handler_context():
                self._invoke_installed(signal.SIGTERM)
        finally:
            signal.signal(signal.SIGTERM, prev)

        self.assertTrue(trainer.should_stop)


if __name__ == "__main__":
    unittest.main()
