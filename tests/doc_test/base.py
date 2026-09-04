"""Markdown document label test base class
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mistune

# ============================================================
# Data structures: one schema per label
# ============================================================


@dataclass(frozen=True)
class SetupCommand:
    cmd: str
    store: str | None
    hidden: bool
    load: tuple = ()  # ((store_var, local_name), ...)

    def __post_init__(self) -> None:
        if not self.cmd:
            raise LabelSpecError(
                'SetupCommand.cmd must be a non-empty string'
            )
        if self.store is not None and not self.store:
            raise LabelSpecError(
                'SetupCommand.store must be None or a non-empty string'
            )
        for i, item in enumerate(self.load):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(x, str) and x for x in item)
            ):
                raise LabelSpecError(
                    f'SetupCommand.load[{i}] must be a (str, str) tuple; '
                    f'got {item!r}'
                )


@dataclass(frozen=True)
class TestCommand:
    id: str
    cmd: str
    language: str
    load: tuple = ()  # ((store_var, local_name), ...)

    def __post_init__(self) -> None:
        if not self.id:
            raise LabelSpecError('TestCommand.id must be non-empty')
        if not self.cmd:
            raise LabelSpecError('TestCommand.cmd must be non-empty')
        if not self.language:
            raise LabelSpecError('TestCommand.language must be non-empty')
        # load is the ((store_var, local_name), ...) shape
        for i, item in enumerate(self.load):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(x, str) and x for x in item)
            ):
                raise LabelSpecError(
                    f'TestCommand.load[{i}] must be a (str, str) tuple; '
                    f'got {item!r}'
                )


@dataclass(frozen=True)
class TestExpectedOutput:
    id: str
    body: str
    fuzzy: tuple = ()  # tuple of placeholder strings; empty means use only the default '...'
    disable_fuzzy: bool = False  # when True, disables all non-greedy matching
    load: tuple = ()  # ((store_var, local_name), ...)

    def __post_init__(self) -> None:
        if not self.id:
            raise LabelSpecError('TestExpectedOutput.id must be non-empty')
        if not self.body:
            raise LabelSpecError('TestExpectedOutput.body must be non-empty')
        for i, p in enumerate(self.fuzzy):
            if not isinstance(p, str) or not p:
                raise LabelSpecError(
                    f'TestExpectedOutput.fuzzy[{i}] must be a non-empty '
                    f'string; got {p!r}'
                )
        if self.disable_fuzzy and self.fuzzy:
            raise LabelSpecError(
                'TestExpectedOutput.disable_fuzzy=True conflicts with '
                f'non-empty fuzzy={self.fuzzy!r}'
            )
        for i, item in enumerate(self.load):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(x, str) and x for x in item)
            ):
                raise LabelSpecError(
                    f'TestExpectedOutput.load[{i}] must be a (str, str) '
                    f'tuple; got {item!r}'
                )


class LabelSpecError(Exception):
    """Contract violation. The error message includes enough context (id / load value / currently-known
    store set) to locate the offending code block directly in the document."""

# ============================================================
# Module-level utilities
# ============================================================


def _rescan_fences(raw: str) -> list[tuple[str, str]]:
    """Carve out all ``` fences from ``block_html.raw``, returning ``[(info, body), ...]``.

    Example:

        Input raw (mistune's ``block_html.raw`` field)::

            <!--
            ```shell #test-setup store="x"
            echo captured
            ```
            some prose
            ```shell #test-setup store="y"
            echo twice
            ```
            -->

        Returns ``[
            ('shell #test-setup store="x"', 'echo captured'),
            ('shell #test-setup store="y"', 'echo twice'),
        ]`` — splits the two fences swallowed inside the comment.

    Unclosed raises ``LabelSpecError`` (keep the contract's error type so doc authors don't see a pile of different exception classes).
    """
    out: list[tuple[str, str]] = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith('```'):
            info = lines[i].lstrip()[3:].strip()
            j = i + 1
            body_lines: list[str] = []
            closed = False
            while j < len(lines):
                if lines[j].lstrip().startswith('```'):
                    out.append((info, '\n'.join(body_lines)))
                    i = j + 1
                    closed = True
                    break
                body_lines.append(lines[j])
                j += 1
            if not closed:
                raise LabelSpecError(
                    f'unclosed fence inside HTML comment: '
                    f'info={info!r} body_head={body_lines[:1]!r}'
                )
        else:
            i += 1
    return out

# ============================================================
# Base class: template method pattern
# ============================================================


_MD_AST = mistune.create_markdown(renderer='ast', plugins=[])


class MarkdownDocTestBase(ABC):
    DEFAULT_COMMAND_TIMEOUT: int = 1800  # 30 minutes; subclasses with long training commands should override.
    USER_AGENT: str = 'markdown-doc-test/1.0'  # subclasses mirroring a monitored source override.
    ERROR_MARKERS: tuple[str, ...] = (
        '[ERROR]',
        'Traceback (most recent call last)',
    )

    # ============================================================
    # Private: parser internals
    # ============================================================

    _LABEL_TEST = '#test'
    _LABEL_TEST_RESULT = '#test-result'
    _LABEL_TEST_SETUP = '#test-setup'
    _KNOWN_LABELS = (_LABEL_TEST, _LABEL_TEST_RESULT, _LABEL_TEST_SETUP)
    _KNOWN_PARAMS = frozenset({'id', 'store', 'load', 'fuzzy', 'disable_fuzzy'})
    _DEFAULT_FUZZY_PLACEHOLDER = '...'
    _KNOWN_LANGUAGES = ('shell',)

    _FLAG_PARAMS = ('disable_fuzzy',)

    @staticmethod
    def _parse_params(param_strs: list[str]) -> dict[str, list[str]]:
        """Parse ``key='value'`` / ``key="value"`` tokens into a multi-value dict.

        Value-less flags (those in ``_FLAG_PARAMS``, e.g. ``disable_fuzzy``) are accepted
        only in their bare form — ``disable_fuzzy='false'`` is rejected. The consumer
        (``_parse_block``) reads the flag's presence via ``bool(params.get(key))`` and
        ignores any list contents, so a quoted ``'false'`` would silently flip semantics
        from "the author wanted to disable" to "flag is set". Failing here keeps the
        contract's "no value" promise visible at parse time.
        """
        params: dict[str, list[str]] = {}
        for tok in param_strs:
            if '=' not in tok:
                if tok in MarkdownDocTestBase._FLAG_PARAMS:
                    params.setdefault(tok, ['1'])
                    continue
                raise LabelSpecError(
                    f"invalid parameter (no '='): {tok!r}"
                )
            key, _, value = tok.partition('=')
            # Flags are contractually value-less; reject ``flag='x'`` so the author's
            # intent isn't silently inverted by ``bool(non-empty list) == True``.
            if key in MarkdownDocTestBase._FLAG_PARAMS:
                raise LabelSpecError(
                    f"flag parameter {key!r} takes no value; write it "
                    f"bare, got {tok!r}"
                )
            # len(value) < 2 means only quotes, no content
            if len(value) < 2 or not (
                (value.startswith("'") and value.endswith("'"))
                or (value.startswith('"') and value.endswith('"'))
            ):
                raise LabelSpecError(
                    f'parameter value must be single- or double-quoted: {tok!r}'
                )
            params.setdefault(key, []).append(value[1:-1])
        return params

    @staticmethod
    def _parse_load_value(value: str) -> tuple[str, str]:
        """``xxx>>yyy`` -> ``(xxx, yyy)``。"""
        if '>>' not in value:
            raise LabelSpecError(
                f"load= value must be in xxx>>yyy form: {value!r}"
            )
        parts = value.split('>>')
        if len(parts) != 2 or not all(parts):
            raise LabelSpecError(
                f"load= value must be exactly 'store>>placeholder': {value!r}"
            )
        return parts[0], parts[1]
    
    def _scan_blocks(self, text: str) -> list[dict]:
        """Identify code blocks, or code blocks inside HTML comments (<!-- -->)
        """
        ast: Any = _MD_AST(text)
        blocks: list[dict] = []
        for node in ast:
            if node['type'] == 'block_html':
                raw = node['raw']
                if not raw.lstrip().startswith('<!--'):
                    continue
                for info, body in _rescan_fences(raw):
                    blocks.append({
                        'info': info,
                        'body': body.rstrip('\n'),
                        'hidden': True,
                    })
            elif node['type'] == 'block_code':
                attrs = node.get('attrs') or {}
                info = attrs.get('info', '') or ''
                blocks.append({
                    'info': info,
                    'body': node['raw'].rstrip('\n'),
                    'hidden': False,
                })
        return blocks

    def _parse_block(self, block: dict) -> dict | None:
        """Parse a single code block's info string. Returns ``None`` for unlabeled (rule 9 skips it)."""
        info = block['info']
        parts = info.split()
        if not parts:
            return None

        label_idx = -1
        label: str | None = None
        for k, p in enumerate(parts):
            if p in self._KNOWN_LABELS:
                label_idx = k
                label = p
                break
        if label_idx < 0:
            return None

        language = parts[0] if label_idx > 0 else None

        param_strs = parts[label_idx + 1:]
        params = self._parse_params(param_strs)

        ids = params.get('id', [])
        if len(ids) > 1:
            raise LabelSpecError(
                'duplicate id= parameter on the same block'
            )
        block_id = ids[0] if ids else None

        stores = params.get('store', [])
        if len(stores) > 1:
            raise LabelSpecError(
                'duplicate store= parameter on the same block'
            )
        store = stores[0] if stores else None
        if store is not None and label != self._LABEL_TEST_SETUP:
            raise LabelSpecError(
                f"store= is only valid on #test-setup, got {label}"
            )

        fuzzies = params.get('fuzzy', [])
        # fuzzy= only allowed on #test-result: #test is the command body, #test-setup is the setup
        # command; body doesn't participate in fuzzy matching.
        if fuzzies and label != self._LABEL_TEST_RESULT:
            raise LabelSpecError(
                f"fuzzy= is only valid on #test-result, got {label}"
            )
        # Duplicate placeholder is a contract violation: usually a typo.
        if len(fuzzies) != len(set(fuzzies)):
            raise LabelSpecError(
                f'duplicate fuzzy placeholder: {fuzzies!r}'
            )

        disable_fuzzy = bool(params.get('disable_fuzzy'))
        if disable_fuzzy and fuzzies:
            raise LabelSpecError(
                "disable_fuzzy conflicts with fuzzy=: pick one"
            )
        if disable_fuzzy and label != self._LABEL_TEST_RESULT:
            raise LabelSpecError(
                f"disable_fuzzy is only valid on #test-result, got {label}"
            )

        loads: list[tuple[str, str]] = []
        for raw in params.get('load', []):
            loads.append(self._parse_load_value(raw))

        unknown = set(params) - self._KNOWN_PARAMS
        if unknown:
            raise LabelSpecError(
                f'unknown parameter(s): {sorted(unknown)}'
            )

        if (
            label == self._LABEL_TEST_RESULT
            and not fuzzies
            and not disable_fuzzy
        ):
            fuzzies = [self._DEFAULT_FUZZY_PLACEHOLDER]

        return {
            'label': label,
            'id': block_id,
            'language': language,
            'load': tuple(loads),
            'store': store,
            'fuzzy': tuple(fuzzies),
            'disable_fuzzy': disable_fuzzy,
            'body': block['body'],
            'hidden': block['hidden'],
        }

    def _validate(self, parsed: list[dict]) -> None:
        """Rules 2/5/7/10/11 validation. Any violation raises ``LabelSpecError``."""
        for p in parsed:
            if p['hidden'] and p['label'] != self._LABEL_TEST_SETUP:
                raise LabelSpecError(
                    f'HTML comment can only contain #test-setup, '
                    f'got {p["label"]}'
                )

        for p in parsed:
            if p['label'] not in (self._LABEL_TEST, self._LABEL_TEST_SETUP):
                continue
            if not p['language']:
                raise LabelSpecError(
                    f"{p['label']} block must specify a language (rule 7); "
                    f'block body={p["body"]!r}'
                )
            if p['language'] not in self._KNOWN_LANGUAGES:
                raise LabelSpecError(
                    f"{p['label']} block language {p['language']!r} not "
                    f"supported (rule 7); supported={self._KNOWN_LANGUAGES}; "
                    f'block body={p["body"]!r}'
                )

        seen_ids: dict[str, set[str]] = {
            label: set() for label in self._KNOWN_LABELS
        }
        for p in parsed:
            if not p['id']:
                continue
            bucket = seen_ids[p['label']]
            if p['id'] in bucket:
                raise LabelSpecError(
                    f"duplicate id {p['id']!r} in {p['label']} blocks"
                )
            bucket.add(p['id'])

        result_ids = {
            p['id'] for p in parsed
            if p['label'] == self._LABEL_TEST_RESULT and p['id']
        }
        for p in parsed:
            if p['label'] == self._LABEL_TEST:
                if not p['id']:
                    raise LabelSpecError('#test block must have id=')
                if p['id'] not in result_ids:
                    raise LabelSpecError(
                        f"#test id={p['id']!r} has no matching #test-result"
                    )
        for p in parsed:
            if p['label'] == self._LABEL_TEST_RESULT and not p['id']:
                raise LabelSpecError('#test-result block must have id=')

        seen_stores: set[str] = set()
        for p in parsed:
            if p['label'] == self._LABEL_TEST_SETUP and p['store']:
                seen_stores.add(p['store'])
            elif p['label'] in (self._LABEL_TEST, self._LABEL_TEST_RESULT):
                for store_var, _local in p['load']:
                    if store_var not in seen_stores:
                        raise LabelSpecError(
                            f"load={store_var!r} references store that "
                            f"hasn't appeared earlier in document "
                            f'(seen_stores so far: {sorted(seen_stores)})'
                        )

    def _fold(
        self, parsed: list[dict]
    ) -> tuple[list, dict]:
        """Generate SetupCommand / TestCommand / TestExpectedOutput by type.

        ``TestExpectedOutput`` goes into a dict for lookup, not in the main sequence.
        """
        commands: list = []
        results: dict = {}
        for p in parsed:
            if p['label'] == self._LABEL_TEST_SETUP:
                commands.append(SetupCommand(
                    cmd=p['body'],
                    store=p['store'],
                    hidden=p['hidden'],
                    load=p['load'],
                ))
            elif p['label'] == self._LABEL_TEST:
                commands.append(TestCommand(
                    id=p['id'],
                    cmd=p['body'],
                    language=p['language'],
                    load=p['load'],
                ))
            elif p['label'] == self._LABEL_TEST_RESULT:
                if p['id'] in results:
                    # Rule 2 already blocks this; defensive here
                    raise LabelSpecError(
                        f"duplicate #test-result id={p['id']!r}"
                    )
                results[p['id']] = TestExpectedOutput(
                    id=p['id'],
                    body=p['body'],
                    fuzzy=p['fuzzy'],
                    disable_fuzzy=p['disable_fuzzy'],
                    load=p['load'],
                )
        return commands, results

    # ============================================================
    # Private: per-step execution details
    # ============================================================

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if isinstance(cmd, SetupCommand):
            actual_cmd = self.substitute_placeholders(
                cmd.cmd, cmd.load, self._captures
            )
            rc, out, err = self.run_command(actual_cmd, env, cwd, timeout)
            if rc != 0:
                raise AssertionError(
                    f'setup command failed (rc={rc}); CMD stderr:\n{err.rstrip() or "(empty)"}'
                )
            if cmd.store:
                self._captures[cmd.store] = out.rstrip()
                self.log(
                    f'[Step {idx}] captured {cmd.store!r} '
                    f'({len(out)}B, stripped)'
                )
            return

        if isinstance(cmd, TestCommand):
            expected_obj = results.get(cmd.id)
            if expected_obj is None:
                # _validate already blocks this; defensive here
                raise AssertionError(
                    f'no #test-result for id={cmd.id!r}'
                )
            actual_cmd = self.substitute_placeholders(
                cmd.cmd, cmd.load, self._captures
            )
            expected_body = self.substitute_placeholders(
                expected_obj.body, expected_obj.load, self._captures
            )
            rc, actual, err = self.run_command(actual_cmd, env, cwd, timeout)
            if rc != 0:
                raise AssertionError(
                    f'test command failed (rc={rc}); CMD stderr:\n{err.rstrip() or "(empty)"}'
                )
            if self.compare_output(
                actual, expected_body,
                fuzzy=expected_obj.fuzzy,
                disable_fuzzy=expected_obj.disable_fuzzy,
            ):
                self.log(f'[Step {idx}] test id={cmd.id!r}: OK')
                self.log_block('expected', expected_body.splitlines())
                self.log_block('actual', actual.splitlines())
                return

            self.log(f'[Step {idx}] test id={cmd.id!r}: MISMATCH')
            self.log_block('expected', expected_body.splitlines(), cap=0)
            self.log_block('actual', actual.splitlines(), cap=0)
            raise AssertionError(
                f'test id={cmd.id!r} output mismatch; see summary above'
            )

        raise AssertionError(f'unknown command type: {type(cmd).__name__}')

    @staticmethod
    def _cmd_label(cmd) -> str:
        if isinstance(cmd, SetupCommand):
            return f'setup store={cmd.store!r}'
        if isinstance(cmd, TestCommand):
            return f'test id={cmd.id!r}'
        return f'unknown:{type(cmd).__name__}'

    # ============================================================
    # Public: template-method entry + subclass hooks + framework implementation
    # ============================================================
    def pre_process(self) -> str:
        """Fetch the doc text from ``MONITORED_DOC_URL``.

        Failures raise ``RuntimeError`` (not ``SkipTest``) so CI fails explicitly. No local
        fallback: stale local copies would drift from the trigger source.
        """
        url = os.environ.get('MONITORED_DOC_URL')
        if not url:
            raise RuntimeError(
                'MONITORED_DOC_URL unset - test must run inside the '
                'workflow which sets it; no local fallback by design.'
            )

        last_err: Exception | None = None
        headers = {'User-Agent': self.USER_AGENT}
        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'
        if 'api.github.com' in url:
            headers['Accept'] = 'application/vnd.github.raw'
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read().decode('utf-8')
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                self.log(
                    f'fetch attempt {attempt+1}/2 failed for {url}: {e!r}'
                )
                time.sleep(2)
        raise RuntimeError(
            f'failed to fetch {url} after 2 attempts: {last_err!r}'
        )

    def post_process(self) -> None:
        """Clean up temp files / upload artifacts / close connections."""
        return

    def execute(self, commands: list, results: dict) -> None:
        """Run commands in order. Failures raise a unittest assertion + dump the failing command itself + actual output."""
        self._captures: dict = {}
        env = os.environ.copy()
        cwd = Path.cwd()
        timeout = self.DEFAULT_COMMAND_TIMEOUT

        for i, cmd in enumerate(commands):
            label = self._cmd_label(cmd)
            self.log(
                f'[Step {i}/{len(commands)-1}] {label} timeout={timeout}s'
            )
            try:
                self._run_one(cmd, results, env, cwd, timeout, i)
            except unittest.SkipTest:
                raise
            except Exception as e:
                # On failure, dump the failing command itself + actual output
                self.log(f'[Step {i}] FAILED: {e}')
                self.log_block('cmd', cmd.cmd.splitlines(), cap=0)
                raise

    def run_template(self) -> None:
        """``pre_process`` -> ``parse`` -> ``execute`` -> ``post_process``。

        ``post_process`` runs in a ``finally`` so cleanup runs even if ``execute`` raises.
        Note: ``run_template`` does not handle environment preparation — subclasses handle it themselves.
        """
        text = self.pre_process()
        commands, test_expected_results = self.parse(text)
        self.log(
            f'parsed {len(commands)} commands, '
            f'{len(test_expected_results)} #test-result blocks'
        )
        try:
            self.execute(commands, test_expected_results)
        finally:
            self.post_process()

    def run_command(
        self, cmd: str, env: dict, cwd, timeout: int
    ) -> tuple[int, str, str]:
        """``bash -c`` + forced flush + on error dump all stderr (<= 256 KB).

        On stdout error path dump first 2000 + last 2000 chars; stderr matching any substring in
        ``self.ERROR_MARKERS`` dumps everything (<= 256 KB), since error markers often sit in the
        middle of the traceback. Subclasses extend ``ERROR_MARKERS`` for project-specific signatures.

        On ``subprocess.TimeoutExpired`` the partial stdout/stderr carried on ``e`` is dumped using the same
        rule before re-raising, so a timeout doesn't strand the reader with only a bare traceback.
        """
        self.log(f'CMD start (timeout={timeout}s): {cmd[:2000]}')
        t0 = time.time()
        try:
            proc = subprocess.run(
                ['bash', '-c', cmd],
                env=env,
                cwd=cwd,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - t0
            partial_out = (e.stdout or b'').decode('utf-8', errors='replace')
            partial_err = (e.stderr or b'').decode('utf-8', errors='replace')
            self.log(
                f'CMD TIMEOUT after {elapsed:.1f}s '
                f'(stdout={len(partial_out)}B stderr={len(partial_err)}B); '
                'dumping partial output before re-raising'
            )
            self._dump_command_output(partial_out, partial_err)
            raise
        out = proc.stdout.decode('utf-8', errors='replace')
        err = proc.stderr.decode('utf-8', errors='replace')
        elapsed = time.time() - t0
        self.log(
            f'CMD done in {elapsed:.1f}s rc={proc.returncode} '
            f'(stdout={len(out)}B stderr={len(err)}B)'
        )

        stderr_has_error = any(m in err for m in self.ERROR_MARKERS)
        if proc.returncode != 0 or stderr_has_error \
                or (out.strip() == '' and err.strip() != ''):
            self._dump_command_output(out, err)

        return proc.returncode, out, err

    def _dump_command_output(self, out: str, err: str) -> None:
        """Apply the head / full / tail dump rule to already-decoded stdout/stderr.

        Always dumps — caller is responsible for deciding whether to call this. stderr matching any
        substring in ``self.ERROR_MARKERS`` dumps everything (<= 256 KB) since markers often sit
        mid-traceback; otherwise head + tail of both streams (first/last 2000 chars). No-op when
        both streams are empty.
        """
        if not out and not err:
            return
        stderr_has_error = any(m in err for m in self.ERROR_MARKERS)
        head_err = err[:2000]
        tail_err = err[-2000:] if len(err) > 2000 else ''
        if stderr_has_error and len(err) <= 256_000:
            self.log(
                f'CMD stderr (full, {len(err)}B):\n{err.rstrip()}'
            )
        else:
            if head_err:
                self.log(f'CMD stderr (head):\n{head_err.rstrip()}')
            if tail_err and tail_err != head_err:
                self.log(f'CMD stderr (tail):\n{tail_err.rstrip()}')
        head_out = out[:2000]
        tail_out = out[-2000:] if len(out) > 2000 else ''
        if head_out:
            self.log(f'CMD stdout (head):\n{head_out.rstrip()}')
        if tail_out and tail_out != head_out:
            self.log(f'CMD stdout (tail):\n{tail_out.rstrip()}')

    def parse(self, text: str) -> tuple[list, dict]:
        """Parse + validate.

        Failures raise ``LabelSpecError``; returns ``(main command sequence, TestExpectedOutput registry)``.
        The main sequence only contains ``SetupCommand`` / ``TestCommand``; ``TestExpectedOutput`` goes
        into a dict; the runner executes ``TestCommand`` and looks up the expected by ``id``.
        """
        raw_blocks = self._scan_blocks(text)
        parsed = [self._parse_block(b) for b in raw_blocks]
        parsed = [p for p in parsed if p is not None]
        self._validate(parsed)
        return self._fold(parsed)
    
    def substitute_placeholders(
        self, text: str, load: tuple, captures: dict
    ) -> str:
        """Substitute ``<local>`` with ``captures[store_var]``.

        When ``store_var`` is not in ``captures``, the placeholder is kept verbatim (load-store ordering validates at parse time
        that load references must come after store, so theoretically it always hits; keeping the literal lets bash err rather than
        silently substituting an empty string, which is easier to diagnose.
        """
        for store_var, local in load:
            if store_var in captures:
                text = text.replace(f'<{local}>', captures[store_var])
        return text

    @staticmethod
    def _literal_match(actual: str, expected: str) -> bool:
        """Full-string literal match: \\A...\\Z anchor so actual doesn't pass via
        substring overlap; ``\\n*\\Z`` allows trailing newlines in actual
        (subprocess output always has one) so tests where expected has no
        trailing ``\\n`` still pass."""
        return re.search(
            rf'\A{re.escape(expected)}\n*\Z', actual, re.DOTALL,
        ) is not None

    def compare_output(
        self, actual: str, expected: str,
        fuzzy: str | tuple[str, ...] = (),
        disable_fuzzy: bool = False,
    ) -> bool:
        """``actual`` vs ``expected`` regex match; ``fuzzy`` lists all non-greedy
        placeholders; each occurrence in expected is non-greedy across-line wildcard (uses ``re.DOTALL``).

        Callers must provide the full placeholder set — the runner pulls from ``TestExpectedOutput.fuzzy``
        directly; unit test callers decide what to pass. Common usage:

            fuzzy=('...',)           # not auto-injected here; () means literal match -- the default lives in _parse_block
            fuzzy=('xxx', 'yyy')     # multiple custom placeholders
            disable_fuzzy=True       # disable all non-greedy matching, match literally
        """
        if disable_fuzzy:
            return self._literal_match(actual, expected)
        placeholders: list[str]
        if isinstance(fuzzy, str):
            placeholders = [fuzzy]
        else:
            placeholders = list(fuzzy)
        if not placeholders:
            return self._literal_match(actual, expected)
        sep_pattern = '|'.join(re.escape(p) for p in placeholders)
        parts = re.split(sep_pattern, expected)
        pattern = r'.*?'.join(re.escape(part) for part in parts)
        return re.search(
            rf'\A{pattern}\n*\Z', actual, re.DOTALL
        ) is not None

    def log(self, msg: str) -> None:
        """``[HH:MM:SS.mmm] {msg}``，``flush=True``。"""
        ms = int(time.time() * 1000) % 1000
        ts = time.strftime('%H:%M:%S') + f'.{ms:03d}'
        print(f'[{ts}] {msg}', flush=True)

    def log_block(self, label: str, lines, cap: int = 30) -> None:
        """Block log: OK path ``cap`` lines head+tail; MISMATCH path ``cap=0`` no truncation.

        Each line is prefixed with a line number (``1.`` / ``2.`` / ``154.``) for easy comparison; oversized output dumps the first
        ``cap/2`` + last ``cap/2`` with ``... [N line(s) elided] ...`` in the middle.
        """
        lines = list(lines)
        self.log(f'  --- {label} (head + tail if huge) ---')
        if cap and len(lines) > cap:
            half = cap // 2
            for i, ln in enumerate(lines[:half], 1):
                self.log(f'  {i:>3}. {ln}')
            elided = len(lines) - 2 * half
            self.log(f'  ... [{elided} line(s) elided] ...')
            tail_start = len(lines) - half + 1
            for offset, ln in enumerate(lines[-half:]):
                self.log(f'  {tail_start + offset:>3}. {ln}')
            return
        for i, ln in enumerate(lines, 1):
            self.log(f'  {i:>3}. {ln}')

