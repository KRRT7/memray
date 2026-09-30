import threading
from pathlib import Path

from memray import AllocatorType
from memray import FileReader
from memray import Tracker
from memray._test import MemoryAllocator
from memray._test import set_thread_name
from tests.utils import filter_relevant_allocations
from tests.utils import skip_if_macos

HERE = Path(__file__).parent
TEST_MULTITHREADED_EXTENSION = HERE / "multithreaded_extension"


def allocating_function(allocator, flag_event, wait_event):
    allocator.valloc(1234)
    allocator.free()
    flag_event.set()
    wait_event.wait()
    allocator.valloc(1234)
    allocator.free()


def test_thread_allocations_after_tracker_is_deactivated(tmpdir):
    # GIVEN
    output = Path(tmpdir) / "test.bin"
    wait_event = threading.Event()
    flag_event = threading.Event()
    allocator = MemoryAllocator()

    # WHEN
    with Tracker(output):
        t = threading.Thread(
            target=allocating_function, args=(allocator, flag_event, wait_event)
        )
        t.start()
        flag_event.wait()

    # Keep allocating in the same thread while the tracker is not active
    wait_event.set()
    t.join()

    # THEN
    relevant_records = list(
        filter_relevant_allocations(FileReader(output).get_allocation_records())
    )
    assert len(relevant_records) == 2

    vallocs = [
        record
        for record in relevant_records
        if record.allocator == AllocatorType.VALLOC
    ]
    assert len(vallocs) == 1
    (valloc,) = vallocs
    assert valloc.size == 1234

    frees = [
        record for record in relevant_records if record.allocator == AllocatorType.FREE
    ]
    assert len(frees) == 1


@skip_if_macos
def test_thread_name(tmpdir):
    # GIVEN
    output = Path(tmpdir) / "test.bin"
    allocator = MemoryAllocator()

    def allocating_function():
        set_thread_name("my thread name")
        allocator.valloc(1234)
        allocator.free()

    # WHEN
    with Tracker(output):
        t = threading.Thread(target=allocating_function)
        t.start()
        t.join()

    # THEN
    relevant_records = list(
        filter_relevant_allocations(FileReader(output).get_allocation_records())
    )
    assert len(relevant_records) == 2

    vallocs = [
        record
        for record in relevant_records
        if record.allocator == AllocatorType.VALLOC
    ]
    assert len(vallocs) == 1
    (valloc,) = vallocs
    assert valloc.size == 1234
    assert "my thread name" == valloc.thread_name


def test_setting_python_thread_name(tmpdir):
    # GIVEN
    output = Path(tmpdir) / "test.bin"
    allocator = MemoryAllocator()
    name_set_inside_thread = threading.Event()
    name_set_outside_thread = threading.Event()
    prctl_rc = -1

    def allocating_function():
        allocator.valloc(1234)
        allocator.free()

        threading.current_thread().name = "set inside thread"
        allocator.valloc(1234)
        allocator.free()

        name_set_inside_thread.set()
        name_set_outside_thread.wait()
        allocator.valloc(1234)
        allocator.free()

        nonlocal prctl_rc
        prctl_rc = set_thread_name("set by prctl")
        allocator.valloc(1234)
        allocator.free()

    # WHEN
    with Tracker(output):
        t = threading.Thread(target=allocating_function, name="set before start")
        t.start()
        name_set_inside_thread.wait()
        t.name = "set outside running thread"
        name_set_outside_thread.set()
        t.join()

    # THEN
    expected_names = [
        "set before start",
        "set inside thread",
        "set outside running thread",
        "set by prctl" if prctl_rc == 0 else "set outside running thread",
    ]
    names = [
        rec.thread_name
        for rec in FileReader(output).get_allocation_records()
        if rec.allocator == AllocatorType.VALLOC
    ]
    assert names == expected_names


def test_replacing_initial_stacks_does_not_corrupt_output(tmp_path, capfd):
    """Threads running before tracking starts get a frozen copy of their
    stack, which is replaced (writing pops for any frames already emitted) on
    their first profile event. That write must not race with other threads'
    writes, or the capture file is corrupted.
    """
    import os
    import zlib

    from memray._test import allocate_without_gil_held

    n_blocked, depth, rounds = 96, 300, 40
    data = bytes(range(256)) * 20000

    def blocked(depth, read_fd, write_fd):
        if depth:
            return blocked(depth - 1, read_fd, write_fd)
        # Blocks until released, then allocates without the GIL, emitting
        # the frozen stack. Returning replaces that stack.
        allocate_without_gil_held(write_fd, read_fd)

    def noisy(stop):
        while not stop.is_set():
            zlib.compress(data, 1)  # allocates without the GIL

    for round in range(rounds):
        # GIVEN
        output = tmp_path / f"test{round}.bin"
        pipes = [(os.pipe(), os.pipe()) for _ in range(n_blocked)]
        threads = [
            threading.Thread(target=blocked, args=(depth, go[0], ready[1]))
            for ready, go in pipes
        ]
        for thread in threads:
            thread.start()
        for ready, _ in pipes:
            os.read(ready[0], 1)

        # WHEN
        stop = threading.Event()
        with Tracker(output):
            noise = [threading.Thread(target=noisy, args=(stop,)) for _ in range(8)]
            for thread in noise:
                thread.start()
            for _, go in pipes:
                os.write(go[1], b"x")
            for thread in threads:
                thread.join()
            stop.set()
            for thread in noise:
                thread.join()
        for ready, go in pipes:
            for fd in (*ready, *go):
                os.close(fd)

        # THEN
        # Reading a corrupted file logs an error and stops early, so fewer
        # records may be read back than the header says were written.
        reader = FileReader(output)
        records = list(reader.get_allocation_records())
        assert "Invalid record type" not in capfd.readouterr().err
        assert len(records) == reader.metadata.total_allocations
        vallocs = [
            record
            for record in records
            if record.allocator == AllocatorType.VALLOC and record.size in (1234, 4321)
        ]
        assert len(vallocs) == 2 * n_blocked
