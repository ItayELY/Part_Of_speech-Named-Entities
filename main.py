from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM, RobertaForCausalLM, RobertaConfig
import torch


def cos_sim(A, B, dim, eps=1e-08):
    numerator = torch.mul(A, B).sum(axis=dim, keepdims=True)
    A_l2 = torch.mul(A, A).sum(axis=dim, keepdims=True)
    B_l2 = torch.mul(B, B).sum(axis=dim, keepdims=True)
    denominator = torch.max(torch.sqrt(torch.mul(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator).squeeze()


tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
model = RobertaModel.from_pretrained('roberta-base')
model2 = RobertaForMaskedLM.from_pretrained("roberta-base")

text0 = "I am so <mask>"
text1 = "look to the left"
text2 = "on the left side"
text3 = "he left me all alone"
text4 = "ayo"

encoded = tokenizer.encode(text0)
print(encoded)


inputs = tokenizer(text0, return_tensors="pt")
am_token_index = ((inputs.input_ids == tokenizer.encode("am")[1])[0].nonzero(as_tuple=True))[0]
mask_token_index = ((inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True))[0]
output0 = model(**inputs)
am_vector = output0.last_hidden_state[0][am_token_index]
mask_vector = output0.last_hidden_state[0][mask_token_index]
#print(tokenizer.decode(50264))
with torch.no_grad():
    labels = inputs["input_ids"]
    output = model2(**inputs, labels=labels)

    # mask labels of non-<mask> tokens
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    outputs = model(**inputs, labels=labels)
    logits = model2(**inputs).logits
    pt_predictions = torch.nn.functional.softmax(output.logits, dim=-1)
    print(output)
    print(logits.shape)
    top5mask = torch.topk(logits[0, mask_token_index], 5).indices
    top5mask_prob = torch.topk(pt_predictions[0, mask_token_index], 5).indices
    predicted_token_id = top5mask[0]
    predicted_token_id_prob = top5mask_prob[0]
    print(tokenizer.decode(predicted_token_id))
    print(tokenizer.decode(predicted_token_id_prob))
    top5am = torch.topk(logits[0, am_token_index], 5).indices
    predicted_token_id = top5am[0]
    print(tokenizer.decode(predicted_token_id))
encoded_input1 = tokenizer(text1, return_tensors='pt')
encoded_input2 = tokenizer(text2, return_tensors='pt')
encoded_input3 = tokenizer(text3, return_tensors='pt')
encoded_input4 = tokenizer.tokenize(text4, return_tensors='pt')

output1 = model(**encoded_input1)
output2 = model(**encoded_input2)
output3 = model(**encoded_input3)
left_token_index1 = ((encoded_input1.input_ids == tokenizer.encode("left")[1])[0].nonzero(as_tuple=True))[0]
left_token_index2 = ((encoded_input2.input_ids == tokenizer.encode("left")[1])[0].nonzero(as_tuple=True))[0]
left_token_index3 = ((encoded_input3.input_ids == tokenizer.encode("left")[1])[0].nonzero(as_tuple=True))[0]

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
print("Cosine Similarity high: " + str(cos(output1.last_hidden_state[0][left_token_index1][0],
                                          output2.last_hidden_state[0][left_token_index2][0].T)))
print("Cosine Similarity low: " + str(cos(output1.last_hidden_state[0][left_token_index1][0],
                                          output3.last_hidden_state[0][left_token_index3][0].T)))
print("tokens of 'ayo' : " + str(encoded_input4))


