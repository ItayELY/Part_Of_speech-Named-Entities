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
text3 = "heya"
encoded = tokenizer.encode(text0)
print(encoded)


inputs = tokenizer(text0, return_tensors="pt")
mask_token_index = (inputs.input_ids == tokenizer.encode("am")[1])[0].nonzero(as_tuple=True)[0]
print(mask_token_index)
print(tokenizer.decode(50264))
with torch.no_grad():
    logits = model2(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.encode("am")[1])[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print(tokenizer.decode(predicted_token_id))
    print(logits.shape)
encoded_input1 = tokenizer(text1, return_tensors='pt')

encoded_input2 = tokenizer(text2, return_tensors='pt')
encoded_input3 = tokenizer.tokenize(text3, return_tensors='pt')

#mask_token_index = (encoded_input3.input_ids == tokenizer.encode("heya")[1])[0].nonzero(as_tuple=True)[0]
print(encoded_input3)
output1 = model(**inputs)
output2 = model(**encoded_input2)
print(output1.last_hidden_state[0][2])

matrix1 = output1.last_hidden_state[0][4]
matrix2 = output2.last_hidden_state[0][3]
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
#print("Cosine Similarity: " + str(cos(matrix1, matrix2.T)))


