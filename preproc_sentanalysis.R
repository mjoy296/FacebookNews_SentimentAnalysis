library(keras)
library(stringr)
library(text2vec)
library(Rtsne)
library(ggplot2)
library(ggrepel)
library(rio)
library(readr)
library(magrittr)
library(tidyverse)
require(quanteda)
library(stringr)
library(stopwords)
require(preText)
library(SpeedReader)
library(ggplot2)


#----loading data ---------

#reading in data frmo separate facebook pages
nytimes_data <- read_csv("nytimes_data.csv")
fox_data <- read_csv("fox_data.csv")

#creating Trump column
nytimes_data$trump = ifelse(grepl('Trump', nytimes_data$posts), 'Trump', 'Not_Trump')
nytimes_data %<>% mutate(key = paste0(trump, '_nyt'))

fox_data$trump = ifelse(grepl('Trump', fox_data$posts), 'Trump', 'Not_Trump')
fox_data %<>% mutate(key = paste0(trump, '_fox'))

#adding in news source
nytimes_data$news <- 'nyt'
fox_data$news <- 'fox'

#combining datafiles
data <- rbind(nytimes_data, fox_data)


#removing unnecessary columns 
data$reactions <- NULL
#data$time <- NULL
data['X1'] <- NULL

#adding in ID column
data %<>%
  mutate(id = row_number()) %>% 
  drop_na()


write_csv(data, 'final_data.csv')

#-----creating corpus and dtm ----------

data$posts = gsub(" ?(f|ht)tp(s?)://(.*)[.][a-z]+", "", data$posts)


#creating corpus
corpus <-  corpus(data,
                  docid_field = "id",
                  text_field = "posts")

#creating stopwords list 
stopwords = stopwords("english")
stopwords = c(stopwords, '\n', 'nytimes.com', ' ', 'of', 'the',
              'https://nyti.ms/3aqp2ip', 'foxnews.com', 'u.', ',',
              '.', '-', '(', ')', '"', '|', 'and', 'a', 'of', 'the', 'in', 'a', 'say' , 'much', 'hold',
              'york', 'news', 'new', 'fox', 'u.s','one', 'make', 'can')

#lemmatizing tokens
tokens <- tokens_replace(tokens(corpus), pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)
# then we add n-grams:
#ngram_toks<- tokens_ngrams(tokens, n = 2:4)

#creating dtm from lemmatized toks
dtm <- dfm(tokens,
           remove = stopwords,
           stem = F,
           remove_numbers = TRUE,
           remove_punct = TRUE)


head(dtm@Dimnames$features, n = 100)


top_terms = as.data.frame(topfeatures(dtm, 40))

top_terms %>% 
  head(10) %>% 
  tibble::rownames_to_column("term") %>% 
  ggplot(aes(x =reorder(term, `topfeatures(dtm, 40)`), y = `topfeatures(dtm, 40)`)) +
  geom_bar(stat = 'identity') + coord_flip() + 
  xlab('Count') + ylab("Term")



##--------using pretext with sample---------
data_sample =  data[sample(nrow(data), 2000), ]
#creating corpus
corpus_sample <-  corpus(data_sample,
                         docid_field = "id",
                         text_field = "posts")

#lemmatizing tokens
tokens_sample <- tokens_replace(tokens(corpus_sample), pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)


#creating dtm from lemmatized toks
dtm_sample <- dfm(tokens_sample,
                  remove = stopwords,
                  stem = F,
                  remove_numbers = TRUE,
                  remove_punct = TRUE)


head(dtm_sample@Dimnames$features, n = 100)

topfeatures(dtm_sample, 40)


#generate factorial preprocessing specifications
preprocessed_documents <- factorial_preprocessing(
  corpus_sample,
  use_ngrams = FALSE,
  infrequent_term_threshold = 0.2,
  verbose = FALSE)

# look at the fields in the list object:
names(preprocessed_documents)
# see the different specifications:
head(preprocessed_documents$choices)

rm(fox_data, nytimes_data)

# generate pretext scores:
preText_results <- preText(
  preprocessed_documents,
  distance_method = "cosine",
  num_comparisons = 20,
  verbose = FALSE)

save(preText_results, file = "pretext.RData")

# create a preText plot ranking specifications
preText_score_plot(preText_results)

#save(a, b, c, file = "stuff.RData")
# see which features "matter"
regression_coefficient_plot(preText_results,
                            remove_intercept = TRUE)


#-----------eda--------------

plot.STM(stm_fit,
         type="summary",
         n = 5)


dtm
m <-as.data.frame(dtm)
m



v <- sort(rowSums(m),decreasing=FALSE )
v 


d <- data.frame(word = names(v),freq=v)
head(d, 10)



# Plot the most frequent words
barplot(dtm_d[1:5,]$freq, las = 2, names.arg = dtm_d[1:5,]$word,
        col ="lightgreen", main ="Top 5 most frequent words",
        ylab = "Word frequencies")



#-------Sentiment Analysis ---------------

sentiments <- function(data){
  
  corpus <-  corpus(data,
                    docid_field = "id",
                    text_field = "posts")
  
  toks <- tokens(corpus, remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE,
                 remove_url = T)
  
  coded <- tokens_lookup(
    toks,
    dictionary =  data_dictionary_LSD2015)
  
  # now we make a document_term matrix out of the coded terms:
  dfm_lsd <- dfm(coded)
  # and convert it to a data.frame:
  valences_by_speech <- convert(dfm_lsd, to = "data.frame")
  
  # get sum of term counts
  all_words <- dfm(toks)
  valences_by_speech$total_words <- rowSums(all_words)
  
  # calculate Y&S measure:
  valences_by_speech$valence <- (valences_by_speech$positive/valences_by_speech$total_words) - (valences_by_speech$negative/valences_by_speech$total_words)
  
  valences = merge
  return(valences_by_speech)
  
}

sents = sentiments(data)
full_sents = merge(sents, data, by.x = 'doc_id', by.y = 'id')

full_sents %>% 
  ggplot(aes(x = valence, color = news)) + 
  geom_density(alpha = .5) + 
  ylab('Density') + xlab("Valence Score")


full_sents %>% 
  mutate(key  = paste0(news, '_', trump)) %>% 
  ggplot(aes(x = valence, color = key)) + 
  geom_density(alpha = .5) + 
  ylab('Density') + xlab("Valence Score") +
  scale_color_manual( values = c("red","orange", "blue", "green"))


full_sents %<>% 
  mutate(ratio = negative/positive,
         neg_pos = ifelse(negative > positive, 1, 0),
         pos_neg = ifelse(positive > negative, 1 ,0)) 



ols.valence <- lm(valence ~ trump + news + trump*news, data = full_sents)
summary(ols.valence)

ols.pos <- lm(positive ~ trump + news + trump*news, data = full_sents)
summary(ols.pos)

ols.neg <- lm(negative ~ trump + news + trump*news, data = full_sents)
summary(ols.neg)

stargazer::stargazer(ols.valence, ols.pos, ols.neg, type = "html", out = 'sent.html')


ols.more_neg <- lm(neg_pos ~ trump + news, data = full_sents)
summary(ols.more_neg)

ols.more_pos <- lm(pos_neg ~ trump + news, data = full_sents)
summary(ols.more_pos)

