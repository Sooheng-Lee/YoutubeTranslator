from threading import Event


class CancelledError(Exception):
    pass


class CancellationToken:
    def __init__(self):
        self._event = Event()

    def cancel(self):
        self._event.set()

    def check(self):
        if self._event.is_set():
            raise CancelledError("사용자가 작업을 취소했습니다.")

    @property
    def cancelled(self):
        return self._event.is_set()
