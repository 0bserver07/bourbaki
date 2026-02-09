/**
 * Agent configuration
 */
export interface AgentConfig {
  /** Model to use for LLM calls (e.g., 'gpt-5.2', 'claude-sonnet-4-20250514') */
  model?: string;
  /** Model provider (e.g., 'openai', 'anthropic', 'google', 'ollama') */
  modelProvider?: string;
  /** Maximum agent loop iterations (default: 10) */
  maxIterations?: number;
  /** AbortSignal for cancelling agent execution */
  signal?: AbortSignal;
}

/**
 * Message in conversation history
 */
export interface Message {
  role: 'user' | 'assistant' | 'tool';
  content: string;
}

// ============================================================================
// Agent Events (for real-time streaming UI)
// ============================================================================

/**
 * Agent is processing/thinking
 */
export interface ThinkingEvent {
  type: 'thinking';
  message: string;
}

/**
 * Tool execution started
 */
export interface ToolStartEvent {
  type: 'tool_start';
  tool: string;
  args: Record<string, unknown>;
}

/**
 * Tool execution completed successfully
 */
export interface ToolEndEvent {
  type: 'tool_end';
  tool: string;
  args: Record<string, unknown>;
  result: string;
  duration: number;
}

/**
 * Tool execution failed
 */
export interface ToolErrorEvent {
  type: 'tool_error';
  tool: string;
  error: string;
}

/**
 * Tool call was blocked or warned due to retry limits
 */
export interface ToolLimitEvent {
  type: 'tool_limit';
  tool: string;
  /** Warning message (tool allowed but approaching limit or similar query) */
  warning?: string;
  /** Block reason (tool not allowed) */
  blockReason?: string;
  /** Whether the tool call was blocked */
  blocked: boolean;
}

/**
 * Final answer generation started
 */
export interface AnswerStartEvent {
  type: 'answer_start';
}

/**
 * Agent completed with final result
 */
export interface DoneEvent {
  type: 'done';
  answer: string;
  toolCalls: Array<{ tool: string; args: Record<string, unknown>; result: string }>;
  iterations: number;
}

/**
 * Checkpoint event - agent paused for human review
 */
export interface CheckpointEvent {
  type: 'checkpoint';
  /** Unique checkpoint ID for resume */
  checkpointId: string;
  /** Current iteration when checkpoint was taken */
  iteration: number;
  /** Reason for checkpoint */
  reason: 'interval' | 'stuck' | 'technique_switch' | 'manual' | 'error';
  /** Path where checkpoint was saved */
  filepath: string;
  /** Optional message for the user */
  message?: string;
}

/**
 * Resume event - agent resumed from checkpoint
 */
export interface ResumeEvent {
  type: 'resume';
  /** Checkpoint ID being resumed */
  checkpointId: string;
  /** Iteration count from checkpoint */
  iteration: number;
}

/**
 * Union type for all agent events
 */
export type AgentEvent =
  | ThinkingEvent
  | ToolStartEvent
  | ToolEndEvent
  | ToolErrorEvent
  | ToolLimitEvent
  | AnswerStartEvent
  | DoneEvent
  | CheckpointEvent
  | ResumeEvent;
