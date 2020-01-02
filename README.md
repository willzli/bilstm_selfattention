# bilstm_selfattention
bilstm + selfattention core code (tensorflow 1.12.1 / pytorch 1.1.0) is implemented according to paper “A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING”<br/><br/>
paper: https://arxiv.org/pdf/1703.03130.pdf


注意：tensorflow版实现的较早了，后来组内工作迁移至pytorch，在tensorflow版基础上实现了pytorch版。pytorch版相对tensorflow版增加了如下功能（这些功能只是锦上添花，对效果影响很小）：<br/>
* 可选LSTM或GRU作为RNN单元（tensorflow版写死只能用LSTM）。
* 双向RNN结构可以选择性去除，只用selfattention机制。
* 选择性开启或关闭dropout机制。
* 可使用leaky relu（tensorflow版只能用relu）。
