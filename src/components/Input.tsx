/**
 * Bourbaki - Mathematical Reasoning Agent
 * https://github.com/0bserver07/Bourbaki
 *
 * Input component - CLI text input with cursor navigation and slash command autocomplete
 */
import React, { useEffect, useState, useMemo } from 'react';
import { Box, Text, useInput } from 'ink';

import { colors } from '../theme.js';
import { useTextBuffer } from '../hooks/useTextBuffer.js';
import { cursorHandlers } from '../utils/input-utils.js';
import { CursorText } from './CursorText.js';

// Available slash commands for autocomplete
const SLASH_COMMANDS = [
  { cmd: '/help', desc: 'Show available commands' },
  { cmd: '/model', desc: 'Change model/provider' },
  { cmd: '/sessions', desc: 'List or restore sessions' },
  { cmd: '/restore', desc: 'Restore a session by ID' },
  { cmd: '/new', desc: 'Start new session' },
  { cmd: '/debug', desc: 'Toggle debug panel' },
];

interface InputProps {
  onSubmit: (value: string) => void;
  /** Value from history navigation (null = user typing fresh input) */
  historyValue?: string | null;
  /** Callback when user presses up/down arrow for history navigation */
  onHistoryNavigate?: (direction: 'up' | 'down') => void;
}

export function Input({ onSubmit, historyValue, onHistoryNavigate }: InputProps) {
  const { text, cursorPosition, actions } = useTextBuffer();
  const [autocompleteIndex, setAutocompleteIndex] = useState(-1);

  // Get matching slash commands
  const suggestions = useMemo(() => {
    if (!text.startsWith('/') || text.includes(' ')) return [];
    return SLASH_COMMANDS.filter(c => c.cmd.startsWith(text.toLowerCase()));
  }, [text]);

  // Reset autocomplete when text changes
  useEffect(() => {
    setAutocompleteIndex(-1);
  }, [text]);

  // Update input buffer when history navigation changes
  useEffect(() => {
    if (historyValue === null) {
      // Returned to typing mode - clear input for fresh entry
      actions.clear();
    } else if (historyValue !== undefined) {
      // Navigating history - show the historical message
      actions.setValue(historyValue);
    }
  }, [historyValue]);

  // Handle all input
  useInput((input, key) => {
    const ctx = { text, cursorPosition };

    // Up arrow: move cursor up if not on first line, else history navigation
    if (key.upArrow) {
      const newPos = cursorHandlers.moveUp(ctx);
      if (newPos !== null) {
        actions.moveCursor(newPos);
      } else if (onHistoryNavigate) {
        onHistoryNavigate('up');
      }
      return;
    }

    // Down arrow: move cursor down if not on last line, else history navigation
    if (key.downArrow) {
      const newPos = cursorHandlers.moveDown(ctx);
      if (newPos !== null) {
        actions.moveCursor(newPos);
      } else if (onHistoryNavigate) {
        onHistoryNavigate('down');
      }
      return;
    }

    // Cursor movement - left arrow (plain, no modifiers)
    if (key.leftArrow && !key.ctrl && !key.meta) {
      actions.moveCursor(cursorHandlers.moveLeft(ctx));
      return;
    }

    // Cursor movement - right arrow (plain, no modifiers)
    if (key.rightArrow && !key.ctrl && !key.meta) {
      actions.moveCursor(cursorHandlers.moveRight(ctx));
      return;
    }

    // Ctrl+A - move to beginning of current line
    if (key.ctrl && input === 'a') {
      actions.moveCursor(cursorHandlers.moveToLineStart(ctx));
      return;
    }

    // Ctrl+E - move to end of current line
    if (key.ctrl && input === 'e') {
      actions.moveCursor(cursorHandlers.moveToLineEnd(ctx));
      return;
    }

    // Option+Left (Mac) / Ctrl+Left (Windows) / Alt+B - word backward
    if ((key.meta && key.leftArrow) || (key.ctrl && key.leftArrow) || (key.meta && input === 'b')) {
      actions.moveCursor(cursorHandlers.moveWordBackward(ctx));
      return;
    }

    // Option+Right (Mac) / Ctrl+Right (Windows) / Alt+F - word forward
    if ((key.meta && key.rightArrow) || (key.ctrl && key.rightArrow) || (key.meta && input === 'f')) {
      actions.moveCursor(cursorHandlers.moveWordForward(ctx));
      return;
    }

    // Option+Backspace (Mac) / Ctrl+Backspace (Windows) - delete word backward
    if ((key.meta || key.ctrl) && (key.backspace || key.delete)) {
      actions.deleteWordBackward();
      return;
    }

    // Handle backspace/delete - delete character before cursor
    if (key.backspace || key.delete) {
      actions.deleteBackward();
      return;
    }

    // Shift+Enter - insert newline for multi-line input
    if (key.return && key.shift) {
      actions.insert('\n');
      return;
    }

    // Tab - autocomplete slash commands
    if (key.tab && suggestions.length > 0) {
      const nextIndex = (autocompleteIndex + 1) % suggestions.length;
      setAutocompleteIndex(nextIndex);
      actions.setValue(suggestions[nextIndex].cmd);
      return;
    }

    // Handle submit (plain Enter)
    if (key.return) {
      const val = text.trim();
      if (val) {
        onSubmit(val);
        actions.clear();
        setAutocompleteIndex(-1);
      }
      return;
    }

    // Handle regular character input - insert at cursor position
    if (input && !key.ctrl && !key.meta) {
      actions.insert(input);
    }
  });

  return (
    <Box
      flexDirection="column"
      marginBottom={1}
      borderStyle="single"
      borderColor={colors.mutedDark}
      borderLeft={false}
      borderRight={false}
      width="100%"
    >
      <Box paddingX={1}>
        <Text color={colors.primary} bold>
          {'> '}
        </Text>
        <CursorText text={text} cursorPosition={cursorPosition} />
      </Box>

      {/* Slash command suggestions */}
      {suggestions.length > 0 && text.length > 0 && (
        <Box paddingX={3} flexDirection="row" gap={2}>
          {suggestions.map((s, i) => (
            <Text
              key={s.cmd}
              color={i === autocompleteIndex ? colors.primary : colors.muted}
              backgroundColor={i === autocompleteIndex ? colors.mutedDark : undefined}
            >
              {s.cmd} <Text dimColor>{s.desc}</Text>
            </Text>
          ))}
          <Text dimColor>(Tab to complete)</Text>
        </Box>
      )}
    </Box>
  );
}
