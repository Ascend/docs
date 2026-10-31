快速开始
=========================

.. note::

    阅读本篇前，请确保已按照安装指南准备好昇腾环境及lm-evaluation-harness。

.. warning::

    注意目前昇腾 x lm-evaluation-harness仅支持transformers格式的模型评估。

如果要评估transformers类模型，例如评估`Qwen2-0.5B-Instruct`在`MMLU`上的性能，可以使用如下命令：

.. code-block:: shell
    :linenos:

    # 替换HF域名，方便国内用户进行数据及模型的下载
    export HF_ENDPOINT=https://hf-mirror.com

    lm_eval --model hf \
        --model_args pretrained=Qwen2-0.5B-Instruct \
        --tasks MMLU \
        --device npu:0 \  # 设备类型必须指定为npu
        --batch_size 8


出现以下日志代表评估成功：

.. code-block:: shell
    :linenos:

    ...
    hf (pretrained=Qwen/Qwen2-0.5B-Instruct), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 8
    |                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
    |---------------------------------------|-------|------|-----:|------|---|-----:|---|-----:|
    |mmlu                                   |N/A    |none  |     0|acc   |↑  |0.4336|±  |0.0041|
    |  - abstract_algebra                   |      0|none  |     0|acc   |↑  |0.3300|±  |0.0473|
    |  - anatomy                            |      0|none  |     0|acc   |↑  |0.4667|±  |0.0431|
    |  - astronomy                          |      0|none  |     0|acc   |↑  |0.3947|±  |0.0398|
    |  - business_ethics                    |      0|none  |     0|acc   |↑  |0.5400|±  |0.0501|
    |  - clinical_knowledge                 |      0|none  |     0|acc   |↑  |0.4679|±  |0.0307|
    |  - college_biology                    |      0|none  |     0|acc   |↑  |0.3819|±  |0.0406|
    |  - college_chemistry                  |      0|none  |     0|acc   |↑  |0.2800|±  |0.0451|
    |  - college_computer_science           |      0|none  |     0|acc   |↑  |0.3600|±  |0.0482|
    |  - college_mathematics                |      0|none  |     0|acc   |↑  |0.2700|±  |0.0446|
    |  - college_medicine                   |      0|none  |     0|acc   |↑  |0.4277|±  |0.0377|
    |  - college_physics                    |      0|none  |     0|acc   |↑  |0.2941|±  |0.0453|
    |  - computer_security                  |      0|none  |     0|acc   |↑  |0.5000|±  |0.0503|
    |  - conceptual_physics                 |      0|none  |     0|acc   |↑  |0.3532|±  |0.0312|
    |  - econometrics                       |      0|none  |     0|acc   |↑  |0.3158|±  |0.0437|
    |  - electrical_engineering             |      0|none  |     0|acc   |↑  |0.4897|±  |0.0417|
    |  - elementary_mathematics             |      0|none  |     0|acc   |↑  |0.3519|±  |0.0246|
    |  - formal_logic                       |      0|none  |     0|acc   |↑  |0.2857|±  |0.0404|
    |  - global_facts                       |      0|none  |     0|acc   |↑  |0.2800|±  |0.0451|
    |  - high_school_biology                |      0|none  |     0|acc   |↑  |0.4806|±  |0.0284|
    |  - high_school_chemistry              |      0|none  |     0|acc   |↑  |0.3892|±  |0.0343|
    |  - high_school_computer_science       |      0|none  |     0|acc   |↑  |0.4700|±  |0.0502|
    |  - high_school_european_history       |      0|none  |     0|acc   |↑  |0.5697|±  |0.0387|
    |  - high_school_geography              |      0|none  |     0|acc   |↑  |0.5101|±  |0.0356|
    |  - high_school_government_and_politics|      0|none  |     0|acc   |↑  |0.4922|±  |0.0361|
    |  - high_school_macroeconomics         |      0|none  |     0|acc   |↑  |0.4231|±  |0.0250|
    |  - high_school_mathematics            |      0|none  |     0|acc   |↑  |0.2963|±  |0.0278|
    |  - high_school_microeconomics         |      0|none  |     0|acc   |↑  |0.5000|±  |0.0325|
    |  - high_school_physics                |      0|none  |     0|acc   |↑  |0.2185|±  |0.0337|
    |  - high_school_psychology             |      0|none  |     0|acc   |↑  |0.5725|±  |0.0212|
    |  - high_school_statistics             |      0|none  |     0|acc   |↑  |0.3333|±  |0.0321|
    |  - high_school_us_history             |      0|none  |     0|acc   |↑  |0.5049|±  |0.0351|
    |  - high_school_world_history          |      0|none  |     0|acc   |↑  |0.5823|±  |0.0321|
    |  - human_aging                        |      0|none  |     0|acc   |↑  |0.4574|±  |0.0334|
    |  - human_sexuality                    |      0|none  |     0|acc   |↑  |0.5115|±  |0.0438|
    | - humanities                          |N/A    |none  |     0|acc   |↑  |0.4064|±  |0.0070|
    |  - international_law                  |      0|none  |     0|acc   |↑  |0.6694|±  |0.0429|
    |  - jurisprudence                      |      0|none  |     0|acc   |↑  |0.5185|±  |0.0483|
    |  - logical_fallacies                  |      0|none  |     0|acc   |↑  |0.4724|±  |0.0392|
    |  - machine_learning                   |      0|none  |     0|acc   |↑  |0.3036|±  |0.0436|
    |  - management                         |      0|none  |     0|acc   |↑  |0.6214|±  |0.0480|
    |  - marketing                          |      0|none  |     0|acc   |↑  |0.6624|±  |0.0310|
    |  - medical_genetics                   |      0|none  |     0|acc   |↑  |0.4300|±  |0.0498|
    |  - miscellaneous                      |      0|none  |     0|acc   |↑  |0.5160|±  |0.0179|
    |  - moral_disputes                     |      0|none  |     0|acc   |↑  |0.5376|±  |0.0268|
    |  - moral_scenarios                    |      0|none  |     0|acc   |↑  |0.2425|±  |0.0143|
    |  - nutrition                          |      0|none  |     0|acc   |↑  |0.5327|±  |0.0286|
    | - other                               |N/A    |none  |     0|acc   |↑  |0.4796|±  |0.0088|
    |  - philosophy                         |      0|none  |     0|acc   |↑  |0.4759|±  |0.0284|
    |  - prehistory                         |      0|none  |     0|acc   |↑  |0.4444|±  |0.0276|
    |  - professional_accounting            |      0|none  |     0|acc   |↑  |0.3901|±  |0.0291|
    |  - professional_law                   |      0|none  |     0|acc   |↑  |0.3572|±  |0.0122|
    |  - professional_medicine              |      0|none  |     0|acc   |↑  |0.3676|±  |0.0293|
    |  - professional_psychology            |      0|none  |     0|acc   |↑  |0.4314|±  |0.0200|
    |  - public_relations                   |      0|none  |     0|acc   |↑  |0.5000|±  |0.0479|
    |  - security_studies                   |      0|none  |     0|acc   |↑  |0.4857|±  |0.0320|
    | - social_sciences                     |N/A    |none  |     0|acc   |↑  |0.4953|±  |0.0089|
    |  - sociology                          |      0|none  |     0|acc   |↑  |0.6119|±  |0.0345|
    | - stem                                |N/A    |none  |     0|acc   |↑  |0.3689|±  |0.0085|
    |  - us_foreign_policy                  |      0|none  |     0|acc   |↑  |0.6800|±  |0.0469|
    |  - virology                           |      0|none  |     0|acc   |↑  |0.4157|±  |0.0384|
    |  - world_religions                    |      0|none  |     0|acc   |↑  |0.4912|±  |0.0383|

    |      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
    |------------------|-------|------|-----:|------|---|-----:|---|-----:|
    |mmlu              |N/A    |none  |     0|acc   |↑  |0.4336|±  |0.0041|
    | - humanities     |N/A    |none  |     0|acc   |↑  |0.4064|±  |0.0070|
    | - other          |N/A    |none  |     0|acc   |↑  |0.4796|±  |0.0088|
    | - social_sciences|N/A    |none  |     0|acc   |↑  |0.4953|±  |0.0089|
    | - stem           |N/A    |none  |     0|acc   |↑  |0.3689|±  |0.0085|
    ...
