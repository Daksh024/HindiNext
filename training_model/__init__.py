from finetune import fineTune

ft = fineTune("bert-base-multilingual-cased","../tuneCorpus.txt")
model, tokenizer = ft.download_bert()

data = ft.createData()
train_dataset, train_dataloader = ft.dataLoad(data)

optimizer = ft.optimize()
loss_fn = ft.loss()

