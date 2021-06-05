    # Treina o modelo, de acordo com o último banco de dados usado
    print('INICIADO TRATAMENTO')
    frase_processada = list()
    for opiniao in data['text_pt']:
        nova_frase = list()
        opiniao = opiniao.lower()
        palavras_opiniao = token_pontuacao.tokenize(opiniao)
        for palavra in palavras_opiniao:
            if palavra not in stopWords_pontuacao_acentos:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))
    data['tratamento_1'] = frase_processada

    print('INICIADO TRATAMENTO 2')

    frase_processada = list()
    for opiniao in data['tratamento_1']:
        nova_frase = list()
        palavras_opiniao = token_espaco.tokenize(opiniao)
        for palavra in palavras_opiniao:
            if palavra not in stopWords_pontuacao_acentos:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))
    data['tratamento_2'] = frase_processada

    print('INICIADO TRATAMENTO 3')

    frase_processada = list()
    for opiniao in data['tratamento_2']:
        nova_frase = list()
        palavras_opiniao = token_pontuacao.tokenize(opiniao)
        for palavra in palavras_opiniao:
            if palavra not in stopWords_pontuacao_acentos:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))
    data['tratamento_3'] = frase_processada

    print('INICIADO TRATAMENTO 4')

    frase_processada = list()
    for opiniao in data['tratamento_3']:
        nova_frase = list()
        palavras_opiniao = token_pontuacao.tokenize(opiniao)
        for palavra in palavras_opiniao:
            nova_frase.append(stemmer.stem(palavra))
        frase_processada.append(' '.join(nova_frase))
    data['tratamento_4'] = frase_processada

    print('TREINANDO MODELO')

### OLD MODEL ###

    vetorizar = CountVectorizer(lowercase=False, ngram_range = (1,2))
    palavras = vetorizar.fit_transform(data["tratamento_4"])
    treino, teste, class_treino, class_teste = train_test_split(palavras,data["classificacao"],random_state = 42)
    reg_log = LogisticRegression(max_iter=1000)
    reg_log.fit(treino, class_treino)
    acuracia_tfidf = reg_log.score(teste, class_teste)

    print('TREINAMENTO FINALIZADO')
    print('MY MODEL SCORE: ', acuracia_tfidf)