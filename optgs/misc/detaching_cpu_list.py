# import os
# import torch
# from dataclasses import dataclass, is_dataclass, replace, field
# from typing import TypeVar, Generic, Optional, Any, Callable, Iterable
# from pathlib import Path
# import pickle
# import tempfile
# import shutil
# import atexit
# import signal
# from typing import TypeVar, Generic
#
# T = TypeVar('T')
#
#
# @dataclass
# class DetachingCPUList(list[T]):
#     cache_dir: Optional[Path] = field(default=None)
#     save_serializer: Optional[Callable[[Any, Path], None]] = field(default=None)
#     load_serializer: Optional[Callable[[Path], Any]] = field(default=None)
#     detach_func: Optional[Callable[[Any], Any]] = field(default=None)
#     remove_files_on_delete: bool = field(default=True)
#     verbose: bool = field(default=True)
#
#     _cache: dict = field(init=False, repr=False, default_factory=dict)
#     _fd_map: dict = field(init=False, repr=False, default_factory=dict)
#     _no_cache_set: set = field(init=False, repr=False, default_factory=set)  # Track paths marked as no_cache
#     _cache_dir: Path = field(init=False, repr=False)
#     _tmp_dir_created: bool = field(init=False, repr=False, default=False)
#
#     def __post_init__(self):
#
#         # Set default serializers if not provided
#         if self.save_serializer is None:
#             def _save_pickle(obj, path: Path):
#                 path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(path, "wb") as f:
#                     pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
#             self.save_serializer = _save_pickle
#
#         # Set default deserializers if not provided
#         if self.load_serializer is None:
#             def _load_pickle(source):
#                 if isinstance(source, (str, Path)):
#                     with open(str(source), "rb") as f:
#                         return pickle.load(f)
#                 else:
#                     return pickle.load(source)
#             self.load_serializer = _load_pickle
#
#         # Set default detach_func if not provided
#         if self.detach_func is None:
#             self.detach_func = self._detach_recursive
#
#         if self.cache_dir is None:
#             tmp = tempfile.mkdtemp(prefix="detaching_cpu_list_")
#             self._cache_dir = Path(tmp)
#             self._tmp_dir_created = True
#         else:
#             self._cache_dir = Path(self.cache_dir)
#             self._cache_dir.mkdir(parents=True, exist_ok=True)
#
#         atexit.register(self._cleanup)
#         for sig in (signal.SIGINT, signal.SIGTERM):
#             try:
#                 old = signal.getsignal(sig)
#                 def _handler(signum, frame, _old=old):
#                     self._cleanup()
#                     if callable(_old) and _old not in (signal.SIG_DFL, signal.SIG_IGN):
#                         _old(signum, frame)
#                 signal.signal(sig, _handler)
#             except Exception:
#                 pass
#
#     # --------------------------
#     # Core cleanup
#     # --------------------------
#     def _cleanup(self):
#         # Close all file descriptors
#         for fd in list(self._fd_map.values()):
#             try:
#                 os.close(fd)
#             except Exception:
#                 pass
#         self._fd_map.clear()
#         # Remove on-disk files (non-guaranteed mode)
#         if self.remove_files_on_delete and self._tmp_dir_created and self._cache_dir.exists():
#             shutil.rmtree(self._cache_dir, ignore_errors=True)
#
#     # --------------------------
#     # Save / load helpers
#     # --------------------------
#     def _save_unlinked(self, item: Any):
#         """Save item to disk with guaranteed deletion (fd-based approach)."""
#         tmp = tempfile.NamedTemporaryFile(delete=False, dir=str(self._cache_dir))
#         temp_path = Path(tmp.name)
#         tmp.close()
#
#         assert self.save_serializer is not None, "save_serializer must be defined"
#         self.save_serializer(item, temp_path)
#         fd = os.open(str(temp_path), os.O_RDONLY)
#         os.unlink(str(temp_path))  # unlink immediately - kernel guarantees cleanup on fd close
#         pseudo = Path(f"/proc/self/fd/{fd}")
#         token = pseudo.as_posix()
#         self._fd_map[token] = fd
#         if self.verbose:
#             print(f"Saved item to fd {fd} with path {pseudo}")
#         return pseudo
#
#     def _load_from_fd(self, token: str):
#         """Load from file descriptor path."""
#         fd = self._fd_map.get(token)
#         if fd is None:
#             raise RuntimeError(f"FD {token} not available.")
#         dupfd = os.dup(fd)
#         assert self.load_serializer is not None, "load_serializer must be defined"
#         with os.fdopen(dupfd, "rb") as f:
#             f.seek(0)
#             obj = self.load_serializer(f)
#         if self.verbose:
#             print(f"Loaded item from fd {fd} (token {token})")
#         return obj
#
#     def _load_from_disk(self, path: Path):
#         """Load from regular file path."""
#         assert self.load_serializer is not None, "load_serializer must be defined"
#         if self.verbose:
#             print(f"Loading from disk: {path}")
#         return self.load_serializer(path)
#
#     # --------------------------
#     # Public interface
#     # --------------------------
#     def append(self, item, detach_and_cpu: bool = False, save_to_disk: bool = False, no_cache: bool = False):
#         """
#         Append an item to the list.
#
#         Args:
#             item: The item to append
#             detach_and_cpu: If True, apply detach_func to move tensors to CPU
#             save_to_disk: If True, save to disk using fd-based guaranteed deletion
#             no_cache: If True, never cache this item in memory when accessed (always reload from disk)
#         """
#         # Validate save_to_disk requires detach capability
#         if save_to_disk and not detach_and_cpu:
#             raise ValueError("Cannot save to disk without detach_and_cpu=True")
#
#         if not save_to_disk and no_cache:
#             print("Warning: no_cache=True has no effect when save_to_disk=False")
#
#         if detach_and_cpu and self.detach_func:
#             item = self.detach_func(item)
#
#         if save_to_disk:
#             # Always use fd-based guaranteed deletion
#             p = self._save_unlinked(item)
#             super().append(p)
#
#             # Mark as no_cache if requested
#             if no_cache:
#                 self._no_cache_set.add(p.as_posix())
#         else:
#             super().append(item)
#
#     def insert(self, index: int, item, detach_and_cpu: bool = False, save_to_disk: bool = False, no_cache: bool = False):
#         """
#         Insert an item at a specific index in the list.
#
#         Args:
#             index: The index to insert the item at
#             item: The item to insert
#             detach_and_cpu: If True, apply detach_func to move tensors to CPU
#             save_to_disk: If True, save to disk using fd-based guaranteed deletion
#             no_cache: If True, never cache this item in memory when accessed (always reload from disk)
#         """
#         # Validate save_to_disk requires detach capability
#         if save_to_disk and not detach_and_cpu:
#             raise ValueError("Cannot save to disk without detach_and_cpu=True")
#
#         if detach_and_cpu and self.detach_func:
#             item = self.detach_func(item)
#
#         if save_to_disk:
#             # Always use fd-based guaranteed deletion
#             p = self._save_unlinked(item)
#             super().insert(index, p)
#
#             # Mark as no_cache if requested
#             if no_cache:
#                 self._no_cache_set.add(p.as_posix())
#         else:
#             super().insert(index, item)
#
#     def extend(self, items: Iterable[Any], **kwargs):
#         for it in items:
#             self.append(it, **kwargs)
#
#     def __getitem__(self, index):
#         """
#         Return the item at `index`.
#
#         - If the underlying stored value is a Path, load from disk/fd.
#         - If marked as no_cache, always reload from disk (never cache).
#         - Otherwise, cache after first load and return cached object on subsequent accesses.
#         - If not a Path, return the in-memory value directly.
#         """
#         raw = super().__getitem__(index)
#
#         # If it's not a Path, it's an in-memory object: return directly
#         if not isinstance(raw, Path):
#             return raw
#
#         # it's a Path -> use its string as cache key (works for both normal paths and /proc/self/fd/<fd>)
#         key = raw.as_posix()
#
#         # If in cache, return cached object
#         if key in self._cache:
#             assert key not in self._no_cache_set, "Inconsistent state: item both cached and marked no_cache"
#             return self._cache[key]
#
#         # Always reload from disk/fd, never cache
#         if str(raw).startswith("/proc/self/fd/"):
#             obj = self._load_from_fd(key)
#         else:
#             raise NotImplementedError("Loading from disk is not implemented")
#             # return self._load_from_disk(raw)
#
#         if key in self._no_cache_set:
#             pass  # never cache
#         else:
#             # Cache it permanently (unless marked as no_cache)
#             self._cache[key] = obj
#
#         return obj
#
#     def pop(self, index: int = -1):
#         raw = super().pop(index)
#         # If it was a Path, return the loaded object (and keep the fd open / mapping intact)
#         if isinstance(raw, Path):
#             key = raw.as_posix()
#             # return cached value if present; else load now and cache it
#             if key in self._cache:
#                 return self._cache[key]
#             if str(raw).startswith("/proc/self/fd/"):
#                 obj = self._load_from_fd(key)
#             else:
#                 obj = self._load_from_disk(raw)
#             self._cache[key] = obj
#             return obj
#         return raw
#
#     def clear(self):
#         # do not close fds; keep them open until process exit as requested
#         # keep cache intact if you want (or clear it if you prefer)
#         # here we remove list entries but keep any cached objects and open fds
#         super().clear()
#
#     def __iter__(self):
#         """Iterate over items, loading from disk as needed."""
#         for i in range(len(self)):
#             yield self[i]
#
#     def __del__(self):
#         self._cleanup()
#
#     def _detach_recursive(self, obj):
#         if isinstance(obj, torch.Tensor):
#             return obj.detach().cpu()
#         elif isinstance(obj, dict):
#             return {k: self._detach_recursive(v) for k, v in obj.items()}
#         elif isinstance(obj, (list, tuple)):
#             t = type(obj)
#             return t(self._detach_recursive(x) for x in obj)
#         elif is_dataclass(obj):
#             # Replace fields recursively (returns a new instance)
#             return replace(obj, **{
#                 field.name: self._detach_recursive(getattr(obj, field.name))
#                 for field in obj.__dataclass_fields__.values()
#             })
#         else:
#             return obj


from dataclasses import is_dataclass, replace, dataclass
import torch


@dataclass
class DetachingCPUList(list):
    # TODO Naama: Add back disk saving
    def append(self, item, detach_and_cpu=False, save_to_disk=False, no_cache=False):
        if detach_and_cpu:
            item = self._detach_recursive(item)
        super().append(item)

    def extend(self, iterable, detach_and_cpu=False):
        if detach_and_cpu:
            iterable = (self._detach_recursive(x) for x in iterable)
        super().extend(iterable)

    def insert(self, index, item, detach_and_cpu=False, save_to_disk=False, no_cache=False):
        if detach_and_cpu:
            item = self._detach_recursive(item)
        super().insert(index, item)

    def _detach_recursive(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: self._detach_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(self._detach_recursive(x) for x in obj)
        elif is_dataclass(obj):
            # Replace fields recursively (returns a new instance)
            return replace(obj, **{
                field.name: self._detach_recursive(getattr(obj, field.name))
                for field in obj.__dataclass_fields__.values()
            })
        else:
            return obj
