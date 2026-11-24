import { cap, relative, tail } from '../acp/events'
import { isRecord } from '../acp/json'
import type { Adapter, Mode, Norm } from '../adapter'

// Written from agy 1.4.7's `--output-format stream-json`, captured 2026-10-24: agy publishes
// no types for it (its Python SDK speaks a different protocol), so tests pin these shapes. A tool
// step's output arrives with its DONE from 1.5.9 on (captured 2026-11-15).
type Usage = { input_tokens?: number; output_tokens?: number; cache_read_tokens?: number }
type Step = {
  step_index?: number
  state?: string // ACTIVE, then DONE, or ERROR for a failed tool
  step_type?: string // open vocabulary: user_input, agent_response, tool, system_message, …
  text_delta?: string
  tool_name?: string
  tool_info?: { parameters?: unknown; output?: unknown; error?: { message?: unknown } }
  usage?: Usage
}
type AgyEvent =
  | { event: 'init'; conversation_id?: string; init?: { model?: string; cwd?: string } }
  | { event: 'step_update'; step_update?: Step }
  | { event: 'result'; result?: { status?: string; error?: unknown } }

const THINKING = ['low', 'medium', 'high', 'xhigh', 'max'] as const

function wire(line: string): AgyEvent | undefined {
  try {
    const parsed: unknown = JSON.parse(line)
    return isRecord(parsed) && typeof parsed.event === 'string' ? (parsed as AgyEvent) : undefined
  } catch {
    return undefined
  }
}

const clip = (text: string) => (text.length > 100 ? `${text.slice(0, 100)}…` : text)

// agy's tools by what each does, as seen live, so Claude Code folds them ("Ran 2 commands"); any
// other is a step of its own.
const KINDS: Record<string, string> = { read_file: 'read', write_file: 'edit', run_command: 'execute' }

// A command's output comes after agy's own header ("Created At: …\nCommand exited with code 0.");
// the end of a long one is kept, where an error usually is, and a nonzero exit is named.
function outputOf(kind: string, output: string): { output: string; failed: boolean } {
  const code = /Command exited with code (\d+)/.exec(output)?.[1]
  const failed = code !== undefined && code !== '0'
  if (kind !== 'execute') return { output: cap(output, 2000), failed }
  const said = (output.split('Command output:\n')[1] ?? output).trimEnd()
  return { output: tail(failed ? `${said}${said ? '\n' : ''}exit code ${code}` : said, 2000), failed }
}

const flags = (choice: { model?: string; thinking?: string }) => [
  ...(choice.model ? ['--model', choice.model] : []),
  ...(choice.thinking ? ['--effort', choice.thinking] : []),
]

// Claude's permission mode as agy flags; the sandbox stays but for bypass. Bypass approves every tool
// itself, so keeping the sandbox would only cost a round trip per escalation, as with Codex: it is
// dropped for --dangerously-skip-permissions. accept-edits and auto name agy's own modes; default and
// dontAsk pass no --mode, so agy runs as its settings hold them.
const MODE_FLAGS: Partial<Record<Mode, readonly string[]>> = {
  bypassPermissions: ['--dangerously-skip-permissions'],
  acceptEdits: ['--sandbox', '--mode', 'accept-edits'],
  auto: ['--sandbox', '--mode', 'auto'],
}
const permission = (mode: Mode): readonly string[] => MODE_FLAGS[mode] ?? ['--sandbox']

export const agy = {
  name: 'agy',
  bin: 'agy',
  about:
    'Antigravity CLI agent: Gemini, Claude and GPT-OSS models through a Google account, own tools (edits files, runs sandboxed commands; none sandboxed in bypass mode; no read-only mode).',
  // One `id<TAB>name` per line; agy says it is fetching on stderr.
  catalogue: { argv: ['agy', 'models'], ids: (stdout: string) => stdout.split('\n').map((line) => line.split('\t')[0]?.trim() ?? '') },
  // Neither --mode plan nor --sandbox is read-only: plan mode ran a command that wrote a file
  // (tested live on 1.4.7), and --sandbox restricts only terminal commands.
  readOnly: 'none',
  thinking: THINKING,
  steer: false,
  defaultModel: false,
  // The prompt is plain stdin. Slash commands are off, or a prompt starting with `/` runs one.
  // The sandbox confines the commands agy runs (credentials such as ~/.kube are refused by the OS,
  // tested live); it does not stop agy's own file edits, and bypass mode drops it (see MODE_FLAGS).
  start: ({ choice, prompt, session, mode }) => ({
    argv: [
      'agy',
      '--output-format',
      'stream-json',
      '--disable-slash-commands',
      ...permission(mode),
      ...flags(choice),
      ...(session ? ['--conversation', session] : []),
    ],
    input: prompt,
  }),
  // `/model` resolves the flags to the exact model, effort variant included, without a turn.
  probe: {
    argv: (choice) => ['agy', '--output-format', 'json', ...flags(choice), '-p=/model'],
    model: (stdout) => {
      try {
        const parsed: unknown = JSON.parse(stdout)
        const id = isRecord(parsed) && isRecord(parsed.command) && isRecord(parsed.command.data) ? parsed.command.data.id : undefined
        return typeof id === 'string' && id ? id : undefined
      } catch {
        return undefined
      }
    },
  },
  events: (session) => {
    let speaking: number | undefined
    let cwd = ''
    const open = new Set<string>()
    return (line): Norm[] => {
      const e = wire(line)
      switch (e?.event) {
        case 'init': {
          const id = e.conversation_id
          if (typeof e.init?.cwd === 'string') cwd = e.init.cwd
          const out: Norm[] = typeof id === 'string' && id ? [{ k: 'session', id }] : []
          // Empty on 1.4.7; claudexor reads it, so a version that fills it is believed over the probe.
          if (typeof e.init?.model === 'string' && e.init.model) out.push({ k: 'model', name: e.init.model })
          // A missing conversation is only a warning: agy starts a new one and succeeds.
          if (session && id && id !== session)
            out.push({
              k: 'notice',
              t: "agy could not find this agent's earlier conversation, so this answer comes from a new one without that context.",
            })
          return out
        }
        case 'step_update': {
          const s = e.step_update
          if (s?.step_type === 'agent_response') {
            const out: Norm[] = []
            // Each agent_response step is one message; only the last is the answer.
            if (s.step_index !== speaking) {
              speaking = s.step_index
              out.push({ k: 'message' })
            }
            if (typeof s.text_delta === 'string' && s.text_delta) out.push({ k: 'text', t: s.text_delta })
            // Per step, never result.usage too: that is the same tokens summed.
            // input_tokens excludes cached reads: a captured step read 17582 cached tokens on 3464 input.
            if (s.state === 'DONE' && s.usage) {
              const u = s.usage
              out.push({
                k: 'usage',
                usage: { input: u.input_tokens ?? 0, cacheRead: u.cache_read_tokens ?? 0, cacheWrite: 0, output: u.output_tokens ?? 0 },
              })
            }
            return out
          }
          // A tool step comes twice, ACTIVE then DONE: a call to via's step tool, then its end.
          if (s?.step_type === 'tool' && typeof s.step_index === 'number') {
            const id = `agy-step-${s.step_index}`
            const tool = String(s.tool_name ?? 'tool')
            const kind = KINDS[tool] ?? 'other'
            const given = isRecord(s.tool_info?.parameters) ? s.tool_info.parameters : {}
            if (s.state === 'ACTIVE') {
              if (open.has(id)) return []
              open.add(id)
              const path = typeof given.Path === 'string' ? given.Path : ''
              const command = typeof given.CommandLine === 'string' ? given.CommandLine : ''
              const title =
                command ||
                (path && kind !== 'other' ? `${kind === 'read' ? 'Read' : 'Write'} ${path}` : `${tool}: ${JSON.stringify(given)}`)
              return [{ k: 'call', id, kind, title: clip(relative(title, cwd)), ...(path.startsWith('/') ? { path } : {}) }]
            }
            // DONE, or ERROR with agy's reason (seen live on 1.5.9). Any other state leaves it open: the
            // run's end closes what is still open.
            if (s.state !== 'DONE' && s.state !== 'ERROR') return []
            if (!open.delete(id)) return []
            const error = s.tool_info?.error?.message
            if (s.state === 'ERROR')
              return [{ k: 'called', id, output: typeof error === 'string' ? error : 'agy reported an error', failed: true }]
            return [{ k: 'called', id, ...outputOf(kind, typeof s.tool_info?.output === 'string' ? s.tool_info.output : '') }]
          }
          // With no step index the call cannot be matched to its end: it stays a text line.
          if (s?.step_type === 'tool' && s.state === 'ACTIVE')
            return [{ k: 'tool', t: `\n▸ ${String(s.tool_name)}: ${clip(JSON.stringify(s.tool_info?.parameters ?? {}) ?? '')}\n` }]
          return []
        }
        case 'result': {
          const status = e.result?.status
          if (status === undefined || status === 'SUCCESS') return []
          const why = e.result?.error
          return [{ k: 'error', message: typeof why === 'string' && why ? why : `agy ended with status ${status}` }]
        }
        default:
          return []
      }
    }
  },
  model: (choice) => choice.model,
} as const satisfies Adapter
