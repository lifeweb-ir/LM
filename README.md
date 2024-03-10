
<div align="center">

[<img src="./assets/logo_en.png">](https://lifewebco.com)

# Lifeweb language models

</div>

Welcome to Lifeweb Language Models repository.
Here we aim to train different Persian Language models and release them publicly in order to contribute our share to the Persian language's AI feild.
First versions of our models are all trained on our own dataset called **Divan** with more than **164 million documents** and more than **10B tokens** which is normalized and deduplicated meticulously to ensure its enrichment and comprehensiveness. A better dataset leads to a better model. 


# Use Models
You can easily access the models using the links of Huggingface model hub provided in the table below.

| Model Name                                         | Base Model | 	Vocabulary Size |  |
|----------------------------------------------------|--|------------------|--|
| [Tehran](https://huggingface.co/lifeweb-ai/tehran) | [Roberta](https://huggingface.co/HooshvareLab/roberta-fa-zwnj-base) | 50000	           |[Results](#Results)|
| [Shiraz](https://huggingface.co/lifeweb-ai/shiraz) |[MobileBert](https://huggingface.co/google/mobilebert-uncased)| 50000            | [Results](#Results)|

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, FillMaskPipeline

model_name = "lifeweb-ai/shiraz"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = "در همین لحظه که شما مشغول [MASK] این متن هستید، میلیون‌ها دیتا در فضای آنلاین در حال تولید است. ما در لایف وب به جمع‌آوری، پردازش و تحلیل این کلان داده (Big Data) می‌پردازیم."


classifier = FillMaskPipeline(model=model, tokenizer=tokenizer)
result = classifier(text)
print(result[0])
#{'score': 0.3584367036819458, 'token': 5764, 'token_str': 'خواندن', 'sequence': 'در همین لحظه که شما مشغول خواندن این متن هستید، میلیون ها دیتا در فضای انلاین در حال تولید است. ما در لایف وب به جمع اوری، پردازش و تحلیل این کلان داده ( big data ) می پردازیم.'}
  ```




# Results

The Lifeweb models are evaluated on three downstream NLP tasks comprising **NER**, **Sentiment Analysis**, and **Emotion Detection** . **Tehran** outperforms every other Persian language model in terms of accuracy and macro F1. Additionally, **Shiraz** is considerably faster, and its accuracy remains highly competitive without compromising much on speed. According to [**MobileBERT paper**](https://arxiv.org/pdf/2004.02984.pdf), this model is 4.3× smaller and 5.5× faster than BERT-base.
We assert that our models outperform all similar models in the field, achieving a new state-of-the-art performance. 

Referencing [**ParsBERT**](https://arxiv.org/abs/2005.12515) and [**AriaBERT**](https://assets.researchsquare.com/files/rs-3558473/v1_covered_d230d5de-50d1-42d5-ba1a-ef400ede52e3.pdf?c=1699474771), we substantiate this claim by demonstrating superior evaluation metrics, even as they themselves have highlighted their better performance among other suitable models. 

Obvious from the table below, you can find the colab codes for each task to use as a tutorial besides the macro F1 score. 

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Model</th>
    <th class="tg-c3ow" colspan="2">NER</th>
    <th class="tg-c3ow" colspan="2">Sentiment</th>
    <th class="tg-c3ow" colspan="1">Emotion</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-c3ow">Arman</td>
    <td class="tg-c3ow">Peyma</td>
    <td class="tg-c3ow"> Sentipers (multi) </td>
    <td class="tg-c3ow"> Snappfood </td>
    <td class="tg-c3ow"> Arman </td>
  </tr>
  <tr>
    <td class="tg-0pky">lifeweb-ai/tehran</td>
    <td class="tg-c3ow"> 72% <br>
    <td class="tg-c3ow"> 91% <br>
    <td class="tg-c3ow"> 64% <br>
    <td class="tg-c3ow"> 89% <br>
    <td class="tg-c3ow"> 76% <br>
  </tr>
  <tr>
    <td class="tg-0pky">lifeweb-ai/shiraz</td>
    <td class="tg-c3ow"> 68% <br><a href="https://colab.research.google.com/drive/15PUAGy9MUSBO3LPdMJ4h9DVKibREv9oY"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 86% <br><a href="https://colab.research.google.com/drive/1lzVsDpl6_WhxsW8mtUNjhXzQPBMNL6Q2"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 59% <br><a href="https://colab.research.google.com/drive/1L87oYYDBY1Fi0GGvjRGSdSk2rZ5vshUV"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 88% <br><a href="https://colab.research.google.com/drive/1-S-VE83IGGGS9lZVydVKa4SnxshFSvT6"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 67% <br><a href="https://colab.research.google.com/drive/12SpUEsOP1I2cCp-gQsifONyu9yDUGuKG"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
  </tr>
  <tr>
    <td class="tg-0pky">HooshvareLab/bert-fa-zwnj-base</td>
    <td class="tg-c3ow"> 67% <br><a href="https://colab.research.google.com/drive/1HApEhtOm2p0ra1NwHLbptaxNeKqXC_TM"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 86% <br><a href="https://colab.research.google.com/drive/1e67UzkbX1HPgayfi8Z1rNNy79AACr1lV"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 60% <br><a href="https://colab.research.google.com/drive/1pub2tq2Qvb08s2w4cE-AfOwzWYXH6rsM"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 88% <br><a href="https://colab.research.google.com/drive/1PyjCTXFB-SXfrG8Bjjpr9py39Q9J8oGZ"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 59% <br><a href="https://colab.research.google.com/drive/13jUeb2694W9SHWNYa1KMbvmeCAhnDpv0"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
  </tr>
  <tr>
    <td class="tg-0pky">HooshvareLab/roberta-fa-zwnj-base</td>
    <td class="tg-c3ow"> 70% <br><a href="https://colab.research.google.com/drive/1a0o6Mx3jlK8ItWdIQgThM81hlSTE6sur"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 86% <br><a href="https://colab.research.google.com/drive/1fMXN5OeWmeLlLnG1gdznvq9ruBmP3UTv"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 56% <br><a href="https://colab.research.google.com/drive/18OzPDKH1mB6-uDVmN0WWZz_etwrsZ_A3"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 87% <br><a href="https://colab.research.google.com/drive/1E-rfJYZmid3a-bEpskU_j_3S4q_SQmGH"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 58% <br><a href="https://colab.research.google.com/drive/1NRphgik9y0fmZP_7MDUjMq6zTP2AfTMj"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
  </tr>
  <tr>
    <td class="tg-0pky">ViraIntelligentDataMining/AriaBERT</td>
    <td class="tg-c3ow"> 69% <br><a href="https://colab.research.google.com/drive/1s0aSjPYntinkupgaAiGZIvwzKXWjNHgA"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 87% <br><a href="https://colab.research.google.com/drive/1qPy0nFHC8bYj9OskUyksF0gQRQ6hRgbT"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 59% <br><a href="https://colab.research.google.com/drive/1P9YaP9Fem5pSlJqPxP2jG2IBq9TsLbaz"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 88% <br><a href="https://colab.research.google.com/drive/1wuGFELbqx0eE1cvmPZRgfklTTa3SkpyW"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"></td>
    <td class="tg-c3ow"> 69% <br><a href="https://colab.research.google.com/drive/1UINarSRMy4yKbSeXKgSUf84IvJh-JC4q"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="" width="87" height="15"></a></td>
  </tr>
</tbody>
</table>

If you tested our models on a public dataset, and you wanted to add your results to the table above, open a pull request or contact us. Also make sure to have your code available online so that we can add a reference.

# Contributors

- Mehrdad Azizi: [**Linkedin**](https://www.linkedin.com/in/mehrdad-azizi-50839489/), [**Github**](https://github.com/mehrazi)
- Reza Salehi Chegeni: [**Linkedin**](https://www.linkedin.com/in/reza-salehi-chegeni-6988ba271/), [**Github**](https://github.com/rezasalehichegeni)
- Parisa Mousavi: [**Linkedin**](https://www.linkedin.com/in/seyede-parisa-mousavi/), [**Github**](https://github.com/Mousavi-Parisa)
- Iman Hashemi: [**Linkedin**](https://www.linkedin.com/in/iman-hashemi-403738a5), [**Github**](https://github.com/hashemiiman)

# Releases

**v1.0(2024-03-09)**

First version of **Tehran** and **Shiraz** models trained on **DIVAN**.

# License

By contributing to this project, you agree that your contributions will be licensed under the [**Apache License 2.0**](https://www.apache.org/licenses/LICENSE-2.0)


