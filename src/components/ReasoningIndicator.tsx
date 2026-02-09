/**
 * Bourbaki - Mathematical Reasoning Agent
 * https://github.com/0bserver07/Bourbaki
 *
 * ReasoningIndicator - Status indicator shown while agent is working on proofs
 */
import React, { useState, useEffect, useRef } from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';
import { colors } from '../theme.js';
import { getRandomThinkingVerb } from '../utils/thinking-verbs.js';

export type ReasoningState =
  | { status: 'idle' }
  | { status: 'thinking' }
  | { status: 'tool'; toolName: string }
  | { status: 'answering'; startTime: number };

// Backward compatibility alias
export type WorkingState = ReasoningState;

interface ReasoningIndicatorProps {
  state: ReasoningState;
}

/**
 * Persistent status indicator shown above the input box while agent is working
 */
export function ReasoningIndicator({ state }: ReasoningIndicatorProps) {
  const [elapsed, setElapsed] = useState(0);
  const [thinkingVerb, setThinkingVerb] = useState(getRandomThinkingVerb);
  const prevStatusRef = useRef<ReasoningState['status']>('idle');

  // Pick a new random verb when transitioning into thinking/tool state
  useEffect(() => {
    const isThinking = state.status === 'thinking' || state.status === 'tool';
    const wasThinking = prevStatusRef.current === 'thinking' || prevStatusRef.current === 'tool';

    if (isThinking && !wasThinking) {
      setThinkingVerb(getRandomThinkingVerb());
    }

    prevStatusRef.current = state.status;
  }, [state.status]);

  // Track elapsed time only when answering
  useEffect(() => {
    if (state.status !== 'answering') {
      setElapsed(0);
      return;
    }

    const startTime = state.startTime;
    setElapsed(Math.floor((Date.now() - startTime) / 1000));

    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [state.status, state.status === 'answering' ? state.startTime : 0]);

  if (state.status === 'idle') {
    return null;
  }

  let statusWord: string;
  let suffixEnd: string;
  switch (state.status) {
    case 'thinking':
    case 'tool':
      statusWord = `${thinkingVerb}...`;
      suffixEnd = ' to interrupt)';
      break;
    case 'answering':
      statusWord = 'Answering';
      suffixEnd = ` to interrupt • ${elapsed}s)`;
      break;
  }

  return (
    <Box>
      <Text color={colors.primary}>
        <Spinner type="dots" />
      </Text>
      <Text color={colors.primary}> </Text>
      <Text color={colors.primary}>{statusWord}</Text>
      <Text color={colors.muted}> (</Text>
      <Text color={colors.muted} bold>esc</Text>
      <Text color={colors.muted}>{suffixEnd}</Text>
    </Box>
  );
}

// Backward compatibility alias
export const WorkingIndicator = ReasoningIndicator;
