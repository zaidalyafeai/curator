export type Message = {
  role: string
  content: string
}

export type DataItem = {
  response: string | Record<string, any>
  request: {
    model: string
    messages: Message[]
  }
  errors: null | any
  row: any
  row_idx: number
  raw_response: {
    id: string
    object: string
    created: number
    model: string
    choices: {
      index: number
      message: {
        role: string
        content: string
        refusal: null | string
      }
      logprobs: null | any
      finish_reason: string
    }[]
    usage: {
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
      prompt_tokens_details: {
        cached_tokens: number
        audio_tokens: number
      }
      completion_tokens_details: {
        reasoning_tokens: number
        audio_tokens: number
        accepted_prediction_tokens: number
        rejected_prediction_tokens: number
      }
    }
    system_fingerprint: string
  }
}

export interface Run {
  run_hash: string;
  dataset_hash: string;
  prompt_func: string;
  model_name: string;
  response_format: string;
  created_time: string;
  last_edited_time: string;
}