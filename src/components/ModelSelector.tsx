/**
 * Bourbaki - Mathematical Reasoning Agent
 * https://github.com/0bserver07/Bourbaki
 *
 * Model and provider selection UI
 */
import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { colors } from '../theme.js';

interface Provider {
  displayName: string;
  providerId: string;
  models: string[];
}

const PROVIDERS: Provider[] = [
  {
    displayName: 'OpenRouter',
    providerId: 'openrouter',
    models: [
      'openrouter/pony-alpha',
      'openrouter/aurora-alpha',
      'arcee-ai/trinity-large-preview:free',
      'stepfun/step-3.5-flash:free',
      'qwen/qwen3-next-80b-a3b-instruct:free',
      'z-ai/glm-4.5-air:free',
      'qwen/qwen3-coder:free',
      'deepseek/deepseek-r1-0528:free',
    ],
  },
  {
    displayName: 'Ollama Cloud',
    providerId: 'ollama-cloud',
    models: [
      'qwen3-coder:480b-cloud',
      'kimi-k2-thinking:cloud',
      'kimi-k2:1t-cloud',
      'deepseek-v3.1:671b-cloud',
      'minimax-m2:cloud',
      'glm-4.6:cloud',
      'qwen3-vl:235b-instruct-cloud',
      'qwen3-vl:235b-cloud',
      'gpt-oss:120b-cloud',
      'gpt-oss:20b-cloud',
    ],
  },
  {
    displayName: 'OpenAI',
    providerId: 'openai',
    models: ['gpt-5.2', 'gpt-4.1'],
  },
  {
    displayName: 'Anthropic',
    providerId: 'anthropic',
    models: ['claude-sonnet-4-5', 'claude-opus-4-5'],
  },
  {
    displayName: 'Google',
    providerId: 'google',
    models: ['gemini-3-flash-preview', 'gemini-3-pro-preview'],
  },
  {
    displayName: 'GLM (Z.AI)',
    providerId: 'glm',
    models: ['glm-4.7', 'glm-4.5-air'],
  },
  {
    displayName: 'xAI',
    providerId: 'xai',
    models: ['grok-4-0709', 'grok-4-1-fast-reasoning'],
  },
  {
    displayName: 'Ollama (Local)',
    providerId: 'ollama',
    models: [], // Populated dynamically from local Ollama API
  },
];

export function getModelsForProvider(providerId: string): string[] {
  const provider = PROVIDERS.find((p) => p.providerId === providerId);
  return provider?.models ?? [];
}

export function getDefaultModelForProvider(providerId: string): string | undefined {
  const models = getModelsForProvider(providerId);
  return models[0];
}

interface ProviderSelectorProps {
  provider?: string;
  onSelect: (providerId: string | null) => void;
}

export function ProviderSelector({ provider, onSelect }: ProviderSelectorProps) {
  const [selectedIndex, setSelectedIndex] = useState(() => {
    if (provider) {
      const idx = PROVIDERS.findIndex((p) => p.providerId === provider);
      return idx >= 0 ? idx : 0;
    }
    return 0;
  });

  useInput((input, key) => {
    if (key.upArrow || input === 'k') {
      setSelectedIndex((prev) => Math.max(0, prev - 1));
    } else if (key.downArrow || input === 'j') {
      setSelectedIndex((prev) => Math.min(PROVIDERS.length - 1, prev + 1));
    } else if (key.return) {
      onSelect(PROVIDERS[selectedIndex].providerId);
    } else if (key.escape) {
      onSelect(null);
    }
  });

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color={colors.primary} bold>
        Select provider
      </Text>
      <Text color={colors.muted}>
        Switch between LLM providers. Applies to this session and future sessions.
      </Text>
      <Box marginTop={1} flexDirection="column">
        {PROVIDERS.map((p, idx) => {
          const isSelected = idx === selectedIndex;
          const isCurrent = provider === p.providerId;
          const prefix = isSelected ? '> ' : '  ';

          return (
            <Text
              key={p.providerId}
              color={isSelected ? colors.primaryLight : colors.primary}
              bold={isSelected}
            >
              {prefix}
              {idx + 1}. {p.displayName}
              {isCurrent ? ' ✓' : ''}
            </Text>
          );
        })}
      </Box>
      <Box marginTop={1}>
        <Text color={colors.muted}>Enter to confirm · esc to exit</Text>
      </Box>
    </Box>
  );
}

interface ModelSelectorProps {
  providerId: string;
  models: string[];
  currentModel?: string;
  onSelect: (modelId: string | null) => void;
}

export function ModelSelector({ providerId, models, currentModel, onSelect }: ModelSelectorProps) {
  // Normalize model names for comparison (strip provider prefix)
  const stripPrefix = (model: string) => {
    const prefixes = ['openrouter:', 'ollama:', 'ollama-cloud:', 'glm:'];
    for (const prefix of prefixes) {
      if (model?.startsWith(prefix)) {
        return model.replace(prefix, '');
      }
    }
    return model;
  };

  const normalizedCurrentModel = stripPrefix(currentModel || '');

  const [selectedIndex, setSelectedIndex] = useState(() => {
    if (normalizedCurrentModel) {
      const idx = models.findIndex((m) => m === normalizedCurrentModel);
      return idx >= 0 ? idx : 0;
    }
    return 0;
  });

  const provider = PROVIDERS.find((p) => p.providerId === providerId);
  const providerName = provider?.displayName ?? providerId;

  useInput((input, key) => {
    if (key.upArrow || input === 'k') {
      setSelectedIndex((prev) => Math.max(0, prev - 1));
    } else if (key.downArrow || input === 'j') {
      setSelectedIndex((prev) => Math.min(models.length - 1, prev + 1));
    } else if (key.return) {
      if (models.length > 0) {
        onSelect(models[selectedIndex]);
      }
    } else if (key.escape) {
      onSelect(null);
    }
  });

  if (models.length === 0) {
    return (
      <Box flexDirection="column" marginTop={1}>
        <Text color={colors.primary} bold>
          Select model for {providerName}
        </Text>
        <Box marginTop={1}>
          <Text color={colors.muted}>No models available. </Text>
          {providerId === 'ollama' && (
            <Text color={colors.muted}>
              Make sure Ollama is running and you have models downloaded.
            </Text>
          )}
        </Box>
        <Box marginTop={1}>
          <Text color={colors.muted}>esc to go back</Text>
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color={colors.primary} bold>
        Select model for {providerName}
      </Text>
      <Box marginTop={1} flexDirection="column">
        {models.map((model, idx) => {
          const isSelected = idx === selectedIndex;
          const isCurrent = normalizedCurrentModel === model;
          const prefix = isSelected ? '> ' : '  ';

          return (
            <Text
              key={model}
              color={isSelected ? colors.primaryLight : colors.primary}
              bold={isSelected}
            >
              {prefix}
              {idx + 1}. {model}
              {isCurrent ? ' ✓' : ''}
            </Text>
          );
        })}
      </Box>
      <Box marginTop={1}>
        <Text color={colors.muted}>Enter to confirm · esc to go back</Text>
      </Box>
    </Box>
  );
}

export { PROVIDERS };
