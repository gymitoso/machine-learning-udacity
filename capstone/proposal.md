# Nanodegree Engenheiro de Machine Learning
## Proposta de projeto final
Gabriel Yan Mitoso

26 de setembro de 2018

## Proposta

### Histórico do assunto
Já não é mais novidade que a criminalidade no Brasil possui níveis acima da média mundial, com níveis particulamente altos para crimes a mão armada e homicídios, onde em sua maioria o narcotráfico está envolvido. Apesar da Constituição Brasileira estabelecer cinco instituições policiais diferentes, todas estão com equipamentos sucateados e déficit em seus efetivos.

Como se não bastasse a alta taxa de criminalidade e as policias sucateadas, a taxa de ocupação dos presídios brasileiros é de 175%, sendo o Brasil, o terceiro país com mais presos no mundo. A imagem a seguir mostra a distribuição dos crimes no sistema federal.

![Distribuição de Crimes](./images/crime-dist.png)

Como cidadão brasileiro que vive com o medo constante de ser assaltado, acredito que o tema é de suma importância.

- http://www.justica.gov.br/news/ha-726-712-pessoas-presas-no-brasil
- http://www.cnmp.mp.br/portal/todas-as-noticias/11314-taxa-de-ocupacao-dos-presidios-brasileiros-e-de-175-mostra-relatorio-dinamico-sistema-prisional-em-numeros
- http://www.ipea.gov.br/atlasviolencia/

### Descrição do problema
Com os dados fornecidos pelo Secretaria Nacional de Segurança Pública, o objetivo do problema é prever a categoria do crime no estado de São Paulo, baseado na cidade, mês, ano e quantidade de ocorrências. Sendo um objetivo também, explorar os dados visualmente, obtendo-se diferentes gráficos.

### Conjuntos de dados e entradas
Os dados que serão utilizados neste problema foram fornecidos pela [Secretaria Nacional de Segurança Pública](http://dados.mj.gov.br/dataset/sistema-nacional-de-estatisticas-de-seguranca-publica). O conjunto de dados trata da contabilização do número de ocorrências registradas, para cada cidade, mês e ano considerado.

Para o escopo deste problema considerei apenas o estado de São Paulo e os anos de 2015 a 2017. A tabela a seguir apresenta a descrição dos dados:

<table>
<th>Atributo</th>
<th>Descrição</th>
<tr>
<td>Código IBGE do Município</td>
<td>Código de identificação do município utilizado pelo IBGE.</td>
</tr>
<tr>
<tr>
<td>Município</td>
<td>Cidade das ocorrências</td>
</tr>
<td>Tipo Crime</td>
<td>Categoria do crime</td>
</tr>
<tr>
<td>Mês</td>
<td>Mês das ocorrências, representados de 1 a 12.</td>
</tr>
<tr>
<td>Ano</td>
<td>Ano das ocorrências</td>
</tr>
<tr>
<td>PC- Qtde de Ocorrências</td>
<td>Quantidade de ocorrências</td>
</tr>
</table>

### Descrição da solução
Neste problema de classificação temos que as ocorrências devem ser classificadas em apenas uma categoria de crime, então utilizar o algoritmo de [K-Nearest Neighbors](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) deve nos trazer bons resultados. Mas analisando as categorias como variáveis alvo, podemos explorar os algoritmos de [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html) e [Regressão Logística](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

### Modelo de referência (benchmark)
_(aproximadamente 1-2 parágrafos)_

Nesta seção, forneça os detalhes de um modelo ou resultado de referência que esteja relacionado ao assunto, definição do problema e solução proposta. Idealmente, o resultado ou modelo de referência contextualiza os métodos existentes ou informações conhecidas sobre o assunto e problema propostos, que podem então ser objetivamente comparados à solução. Descreva detalhadamente como o resultado ou modelo de referência é mensurável (pode ser medido por alguma métrica e claramente observado).

### Métricas de avaliação
_(aprox. 1-2 parágrafos)_

Nesta seção, proponha ao menos uma métrica de avaliação que pode ser usada para quantificar o desempenho tanto do modelo de benchmark como do modelo de solução apresentados. A(s) métrica(s) de avaliação proposta(s) deve(m) ser adequada(s), considerando o contexto dos dados, da definição do problema e da solução pretendida. Descreva como a(s) métrica(s) de avaliação pode(m) ser obtida(s) e forneça um exemplo de representação matemática para ela(s) (se aplicável). Métricas de avaliação complexas devem ser claramente definidas e quantificáveis (podem ser expressas em termos matemáticos ou lógicos)

### Design do projeto
_(aprox. 1 página)_

Nesta seção final, sintetize um fluxo de trabalho teórico para obtenção de uma solução para o problema em questão. Discuta detalhadamente quais estratégias você considera utilizar, quais análises de dados podem ser necessárias de antemão e quais algoritmos serão considerados na sua implementação. O fluxo de trabalho e discussão propostos devem estar alinhados com as seções anteriores. Adicionalmente, você poderá incluir pequenas visualizações, pseudocódigo ou diagramas para auxiliar na descrição do design do projeto, mas não é obrigatório. A discussão deve seguir claramente o fluxo de trabalho proposto para o projeto de conclusão.
