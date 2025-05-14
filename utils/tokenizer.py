
token_to_id = {
    str(i): i for i in range(10)  
}
token_to_id["<start>"] = 10
token_to_id["<finish>"] = 11

id_to_token = {v: k for k, v in token_to_id.items()}

START = token_to_id["<start>"]
FINISH = token_to_id["<finish>"]
# i don't need padding or a pad token because the input is a fixed length sequence of 5

def encode(label_list):
    return [token_to_id[str(d)] for d in label_list]

def decode(token_ids):
    return [id_to_token[t] for t in token_ids]

def prepare_decoder_labels(labels):
    """
    Prepare decoder input and target sequences for training.
    Input labels: [7, 7, 6, 9]
    Output:
        decoder_input  = [<start>, 7, 7, 6, 9]
        decoder_target = [7, 7, 6, 9, <finish>]
    """
    token_ids = encode(labels)
    decoder_input = [START] + token_ids
    decoder_target = token_ids + [FINISH]
    return decoder_input, decoder_target


