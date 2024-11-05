export type Message = {
  role: string
  content: string
}

export type DataItem = [
  {
    model: string
    messages: Message[]
    response_format: {
      type: string
      json_schema: {
        name: string
        schema: any
      }
    }
  },
  {
    id: string
    object: string
    created: number
    model: string
    choices: {
      index: number
      message: {
        role: string
        content: string
        refusal: null
      }
      logprobs: null
      finish_reason: string
    }[]
    usage: {
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
      prompt_tokens_details: {
        cached_tokens: number
      }
      completion_tokens_details: {
        reasoning_tokens: number
      }
    }
    system_fingerprint: string
  },
  number
]