

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{gyptis}[2021/04/21 Gyptis documentation]




\RequirePackage[nobottomtitles*]{titlesec}
\LoadClass[nobib,nohyper]{tufte-book}

\RequirePackage[dvipsnames]{xcolor}


\definecolor{bookred}{HTML}{9f3e34}
\definecolor{lightgray}{HTML}{8b8b8b}
\definecolor{bookbgtitle}{HTML}{f7f5ef}


\RequirePackage{longtable}

% \ifxetex
  \newcommand{\textls}[2][5]{%
    \begingroup\addfontfeatures{LetterSpace=#1}#2\endgroup
  }
  \renewcommand{\allcapsspacing}[1]{\textls[15]{#1}}
  \renewcommand{\smallcapsspacing}[1]{\textls[10]{#1}}
  \renewcommand{\allcaps}[1]{\textls[15]{\MakeTextUppercase{#1}}}
  \renewcommand{\smallcaps}[1]{\smallcapsspacing{\scshape\MakeTextLowercase{#1}}}
  % \renewcommand{\textsc}[1]{\smallcapsspacing{\textsmallcaps{#1}}}
  \renewcommand{\textsc}[1]{\smallcapsspacing{\textsmallcaps{#1}}}
  \RequirePackage{fontspec}
  \setmainfont{Mate}[SmallCapsFont=Mate SC]
  \setsansfont{Encode Sans}[Scale=MatchUppercase]
  \setmonofont{Iosevka Term Light}[Scale=MatchLowercase]
  
  
\newfontfamily{\titlepagefont}{Encode Sans ExtraBold}
  
% \fi
% 
% 


%%
% For nicely typeset tabular material
\RequirePackage{booktabs}



% Bibliography with biblatex and biber
\RequirePackage[
  style=verbose,
  autocite=footnote,
  backend=biber,
  dashed=false,
  hyperref=true,
  refsection=chapter,
  doi=false,isbn=false,url=false,eprint=false,
]{biblatex}
% Define new format that applies a hypertext reference
\DeclareFieldFormat{linked}{%
  \ifboolexpr{ test {\ifhyperref} and not test {\ifentrytype{online}} }
    {\iffieldundef{doi}
       {\iffieldundef{url}
          {\iffieldundef{isbn}
             {\iffieldundef{issn}
                {#1}
                {\href{\worldcatsearch\thefield{issn}}{#1}}}
             {\href{\worldcatsearch\thefield{isbn}}{#1}}}
          {\href{\thefieldfirstword{url}}{#1}}}
       {\href{http://dx.doi.org/\thefield{doi}}{#1}}}
    {#1}}

% URL prefix for WorldCat query
\def\worldcatsearch{http://www.worldcat.org/search?qt=worldcat_org_all&q=}

% Define new command that returns the first word of a given field
\makeatletter
\def\thefieldfirstword#1{%
  \expandafter\expandafter
  \expandafter\firstword
  \expandafter\expandafter
  \expandafter{\csname abx@field@#1\endcsname}}
\def\firstword#1{\firstword@i#1 \@nil}
\def\firstword@i#1 #2\@nil{#1}
\makeatother

% Redefine url format to print only first URL, omit URL prefix
\DeclareFieldFormat{url}{\url{\firstword{#1}}}

\renewbibmacro*{title}{% Based on generic definition from biblatex.def
  \ifboolexpr{ test {\iffieldundef{title}} and test {\iffieldundef{subtitle}} }
    {}
    {\printtext[title]{\printtext[linked]{%
       \printfield[titlecase]{title}%
       \setunit{\subtitlepunct}%
       \printfield[titlecase]{subtitle}}}%
     \newunit}%
  \printfield{titleaddon}}

\renewbibmacro*{periodical}{% Based on generic definition from biblatex.def
  \iffieldundef{title}
    {}
    {\printtext[title]{\printtext[linked]{%
       \printfield[titlecase]{title}%
       \setunit{\subtitlepunct}%
       \printfield[titlecase]{subtitle}}}}}


% Don't link titles in citations
% \AtEveryCite{\DeclareFieldAlias{linked}{default}}


\renewcommand{\cite}[1]{\autocite{#1}}

%%
% For graphics / images
\RequirePackage{graphicx}
\setkeys{Gin}{width=\linewidth,totalheight=\textheight,keepaspectratio}

% 


\RequirePackage{tikz}

% environments.  We use a slightly smaller font.
\RequirePackage{fancyvrb}
\fvset{fontsize=\normalsize}


%%
% Prints a trailing space in a smart way.
\RequirePackage{xspace}

% Generates the index
\RequirePackage{makeidx}
\makeindex



\renewcommand{\maketitlepage}[0]{%
  \cleardoublepage%
  {%
  \sffamily%
  \begin{fullwidth}%
    \begin{tikzpicture}[remember picture,overlay]
       \node[fill=bookred,anchor=south west, minimum height=\paperheight, minimum width=\paperwidth] (names)
         at (current page.south west){};
    \end{tikzpicture}
    \begin{tikzpicture}[remember picture,overlay]
       \node[fill=bookbgtitle,anchor=south west, minimum height=\paperheight-2cm, minimum width=\paperwidth-2cm] (names)
         at ([yshift=1cm,xshift=1cm]current page.south west){};
    \end{tikzpicture}
    
  \fontsize{18}{20}\selectfont\par\noindent\textcolor{lightgray}{\allcaps{\thanklessauthor}}%
  \vspace{11.5pc}%
  \fontsize{36}{40}{\titlepagefont\par\noindent\textcolor{bookred}{\allcaps{\thanklesstitle}}}%
  \vspace{9pc}%
  
  \newcounter{density}
  \setcounter{density}{30}
  \begin{tikzpicture}[remember picture,overlay]
      \def\titlepiccolor{bookred}
      \path[coordinate] (-0.4,-6)  coordinate(A)
                  ++( 90:9cm) coordinate(B)
                  ++(0.:9cm) coordinate(C)
                  ++(-90:9cm) coordinate(D);
      \draw[fill=\titlepiccolor!\thedensity,draw=none] (A) -- (B) -- (C) --(D) -- cycle;
      \foreach \x in {1,...,14}{%
          \pgfmathsetcounter{density}{\thedensity+10}
          \setcounter{density}{\thedensity}
          \path[coordinate] coordinate(X) at (A){};
          \path[coordinate] (A) -- (B) coordinate[pos=.10](A)
                              -- (C) coordinate[pos=.10](B)
                              -- (D) coordinate[pos=.10](C)
                              -- (X) coordinate[pos=.10](D);
          \draw[fill=\titlepiccolor!\thedensity,draw=none] (A)--(B)--(C)-- (D) -- cycle;
      }
  \end{tikzpicture}

  
  
    \vfill%
  \fontsize{14}{16}\selectfont\par\noindent\allcaps{\thanklesspublisher}%
  \end{fullwidth}%
  }
  \thispagestyle{empty}%
  \clearpage%
}


\RequirePackage[unicode,hyperfootnotes=false,xetex]{hyperref}
\hypersetup{}
\hypersetup{
  linkcolor  = bookred,
  citecolor  = bookred,
  urlcolor   = bookred,
  colorlinks = true,
  pdfborder = {0 0 0},
  bookmarksdepth = section,
  pdftitle=\plaintitle,
}


\RequirePackage{etoolbox}% http://ctan.org/pkg/etoolbox

\makeatletter
\patchcmd
  {\@caption}% <cmd>
  {\normalsize}% <search>
  {\@tufte@caption@font\@tufte@caption@justification}% <replace>
  {\typeout{caption patched}}% <success>
  {}% <failure
  
  \makeatother


  
  \makeatletter

  \def\LT@makecaption#1#2#3{%
    \noalign{\smash{\hbox{\kern\textwidth\rlap{\kern\marginparsep
    \parbox[t]{\marginparwidth}{\vspace{10pt}%
  \@tufte@caption@font \@tufte@caption@justification \noindent 
     #1{#2: }\ignorespaces #3}}}}}}
     
     \makeatother
