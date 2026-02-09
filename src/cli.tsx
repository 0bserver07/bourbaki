#!/usr/bin/env bun
/**
 * CLI - Real-time agentic loop interface
 * Shows tool calls and progress in Claude Code style
 */
import React, { useCallback, useRef } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import { config } from 'dotenv';

import { Input } from './components/Input.js';
import { Intro } from './components/Intro.js';
import { ProviderSelector, ModelSelector } from './components/ModelSelector.js';
import { ApiKeyConfirm, ApiKeyInput } from './components/ApiKeyPrompt.js';
import { DebugPanel } from './components/DebugPanel.js';
import { HistoryItemView, WorkingIndicator } from './components/index.js';
import { getApiKeyNameForProvider, getProviderDisplayName } from './utils/env.js';

import { useModelSelection } from './hooks/useModelSelection.js';
import { useAgentRunner } from './hooks/useAgentRunner.js';
import { useInputHistory } from './hooks/useInputHistory.js';

// Load environment variables
config({ quiet: true });

const BACKEND_URL = process.env.BOURBAKI_BACKEND_URL || 'http://localhost:8000';

export function CLI() {
  const { exit } = useApp();

  // Debug mode state
  const [showDebug, setShowDebug] = React.useState(false);

  // Info message state (separate from errors)
  const [infoMessage, setInfoMessage] = React.useState<string | null>(null);

  // Ref to hold setError - avoids TDZ issue since useModelSelection needs to call
  // setError but useAgentRunner (which provides setError) depends on useModelSelection's outputs
  const setErrorRef = useRef<((error: string | null) => void) | null>(null);

  // Model selection state and handlers
  const {
    selectionState,
    provider,
    model,
    startSelection,
    cancelSelection,
    handleProviderSelect,
    handleModelSelect,
    handleApiKeyConfirm,
    handleApiKeySubmit,
    isInSelectionFlow,
  } = useModelSelection((errorMsg) => setErrorRef.current?.(errorMsg));

  // Agent execution state and handlers
  const {
    history,
    workingState,
    error,
    isProcessing,
    sessionId,
    runQuery,
    cancelExecution,
    setError,
    setSessionId,
    loadRestoredHistory,
    clearHistory,
  } = useAgentRunner({ model, modelProvider: provider, maxIterations: 10 });

  // Assign setError to ref so useModelSelection's callback can access it
  setErrorRef.current = setError;

  // Input history for up/down arrow navigation
  const {
    historyValue,
    navigateUp,
    navigateDown,
    saveMessage,
    updateAgentResponse,
    resetNavigation,
  } = useInputHistory();

  // Handle history navigation from Input component
  const handleHistoryNavigate = useCallback((direction: 'up' | 'down') => {
    if (direction === 'up') {
      navigateUp();
    } else {
      navigateDown();
    }
  }, [navigateUp, navigateDown]);

  // Handle user input submission
  const handleSubmit = useCallback(async (query: string) => {
    // Handle exit
    if (query.toLowerCase() === 'exit' || query.toLowerCase() === 'quit') {
      console.log('Goodbye!');
      exit();
      return;
    }

    // Handle model selection command
    if (query === '/model') {
      startSelection();
      return;
    }

    // Handle debug toggle
    if (query === '/debug') {
      setShowDebug(prev => !prev);
      return;
    }

    // Handle help command
    if (query === '/help') {
      setInfoMessage(`Commands:
  /sessions         List saved sessions
  /sessions <id>    Restore a session
  /restore <id>     Restore a session
  /new              Start fresh session
  /model            Change model/provider
  /debug            Toggle debug panel
  /problems         List available problems
  /prove <id>       Start autonomous proof search
  /pause            Pause proof search
  /progress         Show proof search progress
  /skills           List proof techniques
  /export <format>  Export last proof (latex/lean/markdown)
  exit              Quit Bourbaki`);
      return;
    }

    // Handle sessions list command (with optional restore shortcut)
    if (query === '/sessions' || query.startsWith('/sessions ')) {
      const arg = query.slice(9).trim();

      // If an ID was provided, treat as restore
      if (arg) {
        try {
          const res = await fetch(`${BACKEND_URL}/sessions/${arg}`);
          if (!res.ok) {
            setError(`Session '${arg}' not found. Use /sessions to list.`);
            return;
          }
          const session = await res.json();
          const messages = (session.messages || []).map((m: { role: string; content: string }) => ({ role: m.role, content: m.content }));
          setSessionId(session.id);
          loadRestoredHistory(messages);
          const summary = session.summary ? '\n[Context was compacted - summary available]' : '';
          setInfoMessage(`Restored session: ${session.title}\n${messages.length} messages loaded.${summary}`);
        } catch {
          setError(`Failed to reach backend. Is it running at ${BACKEND_URL}?`);
        }
        return;
      }

      // No ID - list sessions
      try {
        const res = await fetch(`${BACKEND_URL}/sessions`);
        const sessions = await res.json();
        if (sessions.length === 0) {
          setInfoMessage('No saved sessions. Start chatting to create one.');
        } else {
          const list = sessions.slice(0, 10).map((s: { id: string; title: string; message_count?: number; messageCount?: number }) =>
            `${s.id}: ${s.title} (${s.message_count ?? s.messageCount ?? '?'} msgs)`
          ).join('\n');
          setInfoMessage(`Recent sessions:\n${list}\n\nUse /sessions <id> or /restore <id> to restore`);
        }
      } catch {
        setError(`Failed to reach backend. Is it running at ${BACKEND_URL}?`);
      }
      return;
    }

    // Handle session restore command
    if (query.startsWith('/restore ')) {
      const restoreId = query.slice(9).trim();
      try {
        const res = await fetch(`${BACKEND_URL}/sessions/${restoreId}`);
        if (!res.ok) {
          setError(`Session '${restoreId}' not found. Use /sessions to list.`);
          return;
        }
        const session = await res.json();
        const messages = (session.messages || []).map((m: { role: string; content: string }) => ({ role: m.role, content: m.content }));
        setSessionId(session.id);
        loadRestoredHistory(messages);
        const summary = session.summary ? '\n[Context was compacted - summary available]' : '';
        setInfoMessage(`Restored session: ${session.title}\n${messages.length} messages loaded.${summary}`);
      } catch {
        setError(`Failed to reach backend. Is it running at ${BACKEND_URL}?`);
      }
      return;
    }

    // Handle new session command
    if (query === '/new') {
      try {
        // model already has provider prefix (e.g. "openrouter:deepseek/...")
        const modelStr = model.includes(':') ? model : `${provider}:${model}`;
        const res = await fetch(`${BACKEND_URL}/sessions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: modelStr }),
        });
        const data = await res.json();
        setSessionId(data.id);
        clearHistory();
        setInfoMessage('Started new session.');
      } catch {
        setError(`Failed to reach backend. Is it running at ${BACKEND_URL}?`);
      }
      return;
    }

    // Handle problems list
    if (query === '/problems') {
      try {
        const res = await fetch(`${BACKEND_URL}/problems`);
        const problems = await res.json();
        if (!problems.length) {
          setInfoMessage('No problems in database.');
        } else {
          const list = problems.map((p: { id: string; title: string; difficulty?: string; domain?: string }) =>
            `  ${p.id}: ${p.title}${p.difficulty ? ` [${p.difficulty}]` : ''}${p.domain ? ` (${p.domain})` : ''}`
          ).join('\n');
          setInfoMessage(`Problems:\n${list}\n\nUse /prove <id> to start autonomous search`);
        }
      } catch {
        setError(`Failed to reach backend. Is it running at ${BACKEND_URL}?`);
      }
      return;
    }

    // Handle autonomous proof search
    if (query.startsWith('/prove ')) {
      const problemId = query.slice(7).trim();
      try {
        // Fetch the problem first
        const pRes = await fetch(`${BACKEND_URL}/problems/${problemId}`);
        if (!pRes.ok) {
          setError(`Problem '${problemId}' not found. Use /problems to list.`);
          return;
        }
        const problem = await pRes.json();
        setInfoMessage(`Starting autonomous proof search for: ${problem.title}\nStrategies will be tried automatically. Use /progress to check status, /pause to stop.`);

        // Start the search
        const res = await fetch(`${BACKEND_URL}/autonomous/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ problem, max_iterations: 50, max_hours: 1.0 }),
        });
        const progress = await res.json();
        setInfoMessage(
          `Proof search started for: ${problem.title}\n` +
          `Session: ${progress.session_id || progress.sessionId || '?'}\n` +
          `Status: ${progress.status}\n` +
          `Use /progress to monitor, /pause to stop.`
        );
      } catch {
        setError(`Failed to start proof search. Is the backend running at ${BACKEND_URL}?`);
      }
      return;
    }

    // Handle pause
    if (query === '/pause') {
      try {
        await fetch(`${BACKEND_URL}/autonomous/pause`, { method: 'POST' });
        setInfoMessage('Proof search paused.');
      } catch {
        setError(`Failed to reach backend. Is it running at ${BACKEND_URL}?`);
      }
      return;
    }

    // Handle progress
    if (query === '/progress') {
      try {
        const res = await fetch(`${BACKEND_URL}/autonomous/progress`);
        const p = await res.json();
        const lines = [
          `Status: ${p.status}`,
          `Problem: ${p.problem_id || p.problemId || '?'}`,
          `Iteration: ${p.iteration}/${p.max_iterations || p.maxIterations || '?'}`,
          `Elapsed: ${Math.round(p.elapsed_seconds || p.elapsedSeconds || 0)}s`,
          `Strategies tried: ${p.strategies_tried || p.strategiesTried || 0}`,
          `Dead ends: ${p.dead_ends || p.deadEnds || 0}`,
          `Proof found: ${p.proof_found || p.proofFound ? 'YES' : 'no'}`,
        ];
        if (p.current_strategy || p.currentStrategy) {
          lines.push(`Current strategy: ${p.current_strategy || p.currentStrategy}`);
        }
        if ((p.insights || []).length > 0) {
          lines.push(`\nInsights:`);
          for (const insight of p.insights.slice(-5)) {
            lines.push(`  - ${insight}`);
          }
        }
        setInfoMessage(lines.join('\n'));
      } catch {
        setError(`Failed to reach backend. Is it running at ${BACKEND_URL}?`);
      }
      return;
    }

    // Handle skills list
    if (query === '/skills') {
      try {
        const res = await fetch(`${BACKEND_URL}/skills`);
        if (res.ok) {
          const skills = await res.json();
          const list = skills.map((s: { name: string; description?: string }) =>
            `  ${s.name}${s.description ? ` — ${s.description}` : ''}`
          ).join('\n');
          setInfoMessage(`Proof Techniques (Skills):\n${list}\n\nThe agent loads these automatically when needed.`);
        } else {
          // Fallback: skills endpoint might not exist yet
          setInfoMessage(`Proof Techniques (21 skills):\n  Basic: proof-by-induction, strong-induction, direct-proof, proof-by-contradiction, pigeonhole-argument, counting-argument\n  Analysis: epsilon-delta-proof, convergence-test, sequence-limit, inequality-chain\n  Geometry: coordinate-proof, synthetic-construction, transformation-proof\n  Algebra: group-homomorphism, ring-ideal-proof, polynomial-proof\n  Advanced: extremal-argument, probabilistic-method, conjecture-exploration, formalize-informal-proof, explain-proof\n\nThe agent loads these automatically when needed.`);
        }
      } catch {
        // Offline fallback
        setInfoMessage(`Proof Techniques (21 skills):\n  Basic: proof-by-induction, strong-induction, direct-proof, proof-by-contradiction, pigeonhole-argument, counting-argument\n  Analysis: epsilon-delta-proof, convergence-test, sequence-limit, inequality-chain\n  Geometry: coordinate-proof, synthetic-construction, transformation-proof\n  Algebra: group-homomorphism, ring-ideal-proof, polynomial-proof\n  Advanced: extremal-argument, probabilistic-method, conjecture-exploration, formalize-informal-proof, explain-proof\n\nThe agent loads these automatically when needed.`);
      }
      return;
    }

    // Handle export
    if (query.startsWith('/export')) {
      const format = query.slice(7).trim() || 'latex';
      // Get last assistant answer from history
      const lastItem = [...history].reverse().find(h => h.answer);
      if (!lastItem) {
        setError('No proof to export. Run a query first.');
        return;
      }
      try {
        const res = await fetch(`${BACKEND_URL}/export`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: lastItem.query || 'Proof',
            statement: lastItem.query || '',
            proof: lastItem.answer || '',
            format,
          }),
        });
        const data = await res.json();
        setInfoMessage(`Exported as ${data.format} (${data.filename}):\n\n${data.content.slice(0, 500)}${data.content.length > 500 ? '\n...' : ''}`);
      } catch {
        setError(`Failed to export. Is the backend running at ${BACKEND_URL}?`);
      }
      return;
    }

    // Ignore if not idle (processing or in selection flow)
    if (isInSelectionFlow() || workingState.status !== 'idle') return;

    // Clear any previous messages
    setInfoMessage(null);

    // Save user message to history immediately and reset navigation
    await saveMessage(query);
    resetNavigation();

    // Run query and save agent response when complete
    const result = await runQuery(query);
    if (result?.answer) {
      await updateAgentResponse(result.answer);
    }
  }, [exit, startSelection, isInSelectionFlow, workingState.status, runQuery, saveMessage, updateAgentResponse, resetNavigation, provider, model, setSessionId, loadRestoredHistory, clearHistory, setError]);

  // Handle keyboard shortcuts
  useInput((input, key) => {
    // Escape key - cancel selection flows or running agent
    if (key.escape) {
      if (isInSelectionFlow()) {
        cancelSelection();
        return;
      }
      if (isProcessing) {
        cancelExecution();
        return;
      }
    }

    // Ctrl+C - cancel or exit
    if (key.ctrl && input === 'c') {
      if (isInSelectionFlow()) {
        cancelSelection();
      } else if (isProcessing) {
        cancelExecution();
      } else {
        console.log('\nGoodbye!');
        exit();
      }
    }
  });

  // Render selection screens
  const { appState, pendingProvider, pendingModels } = selectionState;

  if (appState === 'provider_select') {
    return (
      <Box flexDirection="column">
        <ProviderSelector provider={provider} onSelect={handleProviderSelect} />
      </Box>
    );
  }

  if (appState === 'model_select' && pendingProvider) {
    return (
      <Box flexDirection="column">
        <ModelSelector
          providerId={pendingProvider}
          models={pendingModels}
          currentModel={provider === pendingProvider ? model : undefined}
          onSelect={handleModelSelect}
        />
      </Box>
    );
  }

  if (appState === 'api_key_confirm' && pendingProvider) {
    return (
      <Box flexDirection="column">
        <ApiKeyConfirm
          providerName={getProviderDisplayName(pendingProvider)}
          onConfirm={handleApiKeyConfirm}
        />
      </Box>
    );
  }

  if (appState === 'api_key_input' && pendingProvider) {
    const apiKeyName = getApiKeyNameForProvider(pendingProvider) || '';
    return (
      <Box flexDirection="column">
        <ApiKeyInput
          providerName={getProviderDisplayName(pendingProvider)}
          apiKeyName={apiKeyName}
          onSubmit={handleApiKeySubmit}
        />
      </Box>
    );
  }

  // Main chat interface
  return (
    <Box flexDirection="column">
      <Intro provider={provider} model={model} />

      {/* All history items (queries, events, answers) */}
      {history.map(item => (
        <HistoryItemView key={item.id} item={item} />
      ))}

      {/* Info message display */}
      {infoMessage && (
        <Box marginBottom={1}>
          <Text color="cyan">{infoMessage}</Text>
        </Box>
      )}

      {/* Error display */}
      {error && (
        <Box marginBottom={1}>
          <Text color="red">Error: {error}</Text>
        </Box>
      )}

      {/* Working indicator - only show when processing */}
      {isProcessing && <WorkingIndicator state={workingState} />}

      {/* Input */}
      <Box marginTop={1}>
        <Input
          onSubmit={handleSubmit}
          historyValue={historyValue}
          onHistoryNavigate={handleHistoryNavigate}
        />
      </Box>

      {/* Debug Panel - toggle with /debug command */}
      <DebugPanel maxLines={8} show={showDebug} />
    </Box>
  );
}
