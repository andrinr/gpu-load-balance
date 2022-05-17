\DeclareOption*{\PassOptionsToClass{\CurrentOption}{letter}}
\ProcessOptions

\LoadClass{letter}

\usepackage[utf8]{inputenc}
\usepackage{graphicx}

\renewcommand{\rmdefault}{phv}
\renewcommand{\sfdefault}{phv}

\setlength{\topmargin}{-32mm}
\setlength{\headheight}{19mm}
\setlength{\headsep}{15mm}

\setlength{\textwidth}{145mm}
\setlength{\textheight}{225mm}

\setlength{\footskip}{0mm}
\setlength{\oddsidemargin}{6mm}
\setlength{\evensidemargin}{\oddsidemargin}

\newif\if@domestic

\DeclareOption{domestic}{
  \@domestictrue
}

\DeclareOption{english}{
  \def\IFIref{Ref.:\ }
  \def\@IFIcity{Z\"urich}
  \def\@IFIcountry{Switzerland}
  \def\@IFIuniv{University of Z\"urich}
  \def\@IFIdept{Department of Informatics}
}

\newcommand*{\IFIphone}[1]{\def\IFIphonenum{#1}}
\newcommand*{\IFIemail}[1]{\def\IFImail{#1}}
\newcommand*{\IFIgroup}[1]{\def\IFIfunc{#1}}
\newcommand*{\position}[1]{\def\position{#1}}
\newcommand*{\subject}[1]{\def\@reftxt{#1}}
\newcommand*{\toname}[1]{\def\tonameF{#1}}
\newcommand*{\toaddressstreet}[1]{\def\toAddressS{#1}}
\newcommand*{\toaddresscity}[1]{\def\toAddressC{#1}}
\newcommand*{\toaddresscountry}[1]{\def\toAddressCt{#1}}

\def\dsb{2cm}
\def\dsa{2.0\baselineskip}
\newcommand*{\dateskipbefore}[1]{\def\dsb{#1}}
\newcommand*{\dateskipafter}[1]{\def\dsa{#1}}

\subject{}
\def\@IFIcity{Z\"urich}
\def\@IFIcountry{Schweiz}
\def\@IFIuniv{Universitat Z\"urich}
\def\@IFIdept{Institut f\"ur Informatik}

\def\@IFIlogo{
  \parbox[c]{24mm}{\vspace{6mm}\includegraphics[height=17mm]{unisiegel}}
  \parbox[c]{76mm}{\fontfamily{pfr}\selectfont
  }
}

\def\@IFIlogosmall{
  \parbox[c]{13mm}{\includegraphics[height=11mm]{unisiegel}}
  \parbox[c]{47mm}{\fontfamily{pfr}\selectfont
  }
}

\def\@IFIaddr{
  \parbox[t]{60mm}{
    \fontfamily{phv}\fontsize{10}{0}\selectfont
    \textbf{Department of Informatics}\\
    ~\\\fontsize{7.5}{9}\selectfont
    University of Z\"urich\\
    Department of Informatics\\
    Binzm\"uhlestr.\ 14\\
    CH-8050 \@IFIcity\\
    Phone. \if@domestic\else +41 \fi 44 635 43 11 \\
    Fax  \if@domestic\else +41 \fi 44 635 68 09 \\
    www.ifi.uzh.ch/dbtg\\[2.7ex]
  }
}


\def\IFIref{}

\ProcessOptions\relax

\renewcommand{\stopletter}{\clearpage}

\renewcommand{\ps@firstpage}{
  \setlength{\headheight}{22mm}
  \renewcommand{\@oddhead}{
    \hspace{-59px}\@IFIlogo \hspace{275px}\@IFIaddr \\
  }
  \renewcommand{\@oddfoot}{\@empty}
}

\newcommand{\@smallhead}{
  \hspace*{-20mm}\protect\@IFIlogosmall\hfill\arabic{page}
}

\if@twoside
  \renewcommand{\ps@headings}{
    \renewcommand{\@oddhead}{\@smallhead}
    \renewcommand{\@evenhead}{\@oddhead}
    \renewcommand{\@oddfoot}{\@empty}
    \renewcommand{\@evenfoot}{\@empty}
  }
\else
  \renewcommand{\ps@headings}{
    \renewcommand{\@oddhead}{\@smallhead}
    \renewcommand{\@oddfoot}{\@empty}
  }
\fi

\pagestyle{headings}

\renewcommand*{\opening}[1]{
  \thispagestyle{firstpage}
  \vspace*{0mm}
  \hfill
  \vspace*{18mm}

  \noindent
  \fontsize{7.5}{9}\selectfont
  \begin{tabbing}
  \underline{UZH, Dept. of Informatics, Binzm\"uhlestr. 14,
    CH-8050 Z\"urich\hspace{58px}}
    \= \hspace{31px}
    \= \textbf{\fromaddress} \\
    \>\>   \position    \\
    \tonameF\>\>   Phone \if@domestic\else +41 \fi 44 63\IFIphonenum\\
    \toAddressS\>\>   Fax \if@domestic\else +41 \fi 44 635 68 09\\
    \toAddressC\>\>   \IFImail @ifi.uzh.ch \\
    \toAddressCt\>\>   \\
  \end{tabbing}
  \fontsize{10}{15}\selectfont

  \vspace*{\dsb}
  \@IFIcity, \@date
  \vspace*{\dsa}

  \ifx\@empty\@reftxt
  \else \textbf{\IFIref \@reftxt}\\[2.0\baselineskip]
  \fi
  #1\par\nobreak
}

\renewcommand{\closing}[1]{
  \par\nobreak\medskip
  \stopbreaks
  \noindent
  \ignorespaces #1\\[6\medskipamount]
  \ifx\@empty\fromsig
    \fromname
  \else \fromsig
  \fi\strut
  \par\bigskip
}

\newcommand\newblock{
  \hskip .11em\@plus.33em\@minus.07em
}
  
\let\@openbib@code\@empty

\newenvironment{thebibliography}[1]{
  {\bfseries References}%
  \list{\@biblabel{\@arabic\c@enumiv}}%
  {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth 
    \advance\leftmargin\labelsep \@openbib@code
    \usecounter{enumiv}%
    \let\p@enumiv\@empty 
    \renewcommand\theenumiv{\@arabic\c@enumiv}
  }%
  \sloppy 
  \clubpenalty4000 \@clubpenalty \clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m
} 
  
{\def\@noitemerr {\@latex@warning{Empty
`thebibliography' environment}}%
\endlist}
