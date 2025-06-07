/*

1. Special Token Control: Easily insert/manage start or end tokens based on position.
2. Positional Processing: Apply specific rules (e.g., cleaning, normalization) only to first or last tokens.
3. Debugging Aid: Clearly visualize how tokenization affects word/sequence boundaries.
4. Edge Case Handling: Implement nuanced logic for tokens at the beginning or end of segments.
5. Output Formatting: Customize how first/last tokens appear in final output strings.

*/

// src/tokenizer/first_last_iterator.rs
use std::{iter, mem};

/// Provides access to the `FirstLastIterator` to any Iterator
pub trait WithFirstLastIterator: Iterator + Sized {
    fn with_first_and_last(self) -> FirstLastIterator<Self>;
}

impl<I> WithFirstLastIterator for I
where
    I: Iterator,
{
    fn with_first_and_last(self) -> FirstLastIterator<Self> {
        FirstLastIterator {
            first: true,
            iter: self.peekable(),
        }
    }
}

/// Provides information about whether an item is the first and/or the last of the iterator
pub struct FirstLastIterator<I>
where
    I: Iterator,
{
    first: bool,
    iter: iter::Peekable<I>,
}

impl<I> Iterator for FirstLastIterator<I>
where
    I: Iterator,
{
    /// (is_first, is_last, item)
    type Item = (bool, bool, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let first = mem::replace(&mut self.first, false);
        self.iter
            .next()
            .map(|e| (first, self.iter.peek().is_none(), e))
    }
}