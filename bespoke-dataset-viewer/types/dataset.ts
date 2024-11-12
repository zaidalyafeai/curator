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
  raw_response: null | any
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