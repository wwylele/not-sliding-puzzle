use glam::*;
use rand::*;
use std::time::*;

const PIECE_MOVING_SPEED: f32 = 4.0;
const PIECE_SHAKING_SPEED: f32 = 4.0;
const PIECE_MORPHING_SPEED: f32 = 8.0;

pub struct Moving {
    pub previous_pos: UVec2,
    pub progress: f32,
}

#[derive(Copy, Clone)]
pub enum Kind {
    Normal = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Rule {
    Classic,
    Horsey,
    Full,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BoardSize {
    Small,
    Large,
}

pub struct Piece {
    pub kind: Kind,
    pub source_pos: UVec2,
    pub moving: Option<Moving>,
    pub morph_progress: f32, // 0.0 = show sqaure, 1.0 = show piece kind
    pub shake_progress: f32, // 0.0 = idle, 1.0 = start shaking
}

impl Piece {
    pub fn priority(&self) -> i32 {
        let mut p = 0;
        if self.moving.is_some() {
            p += 1000;
        }
        if self.shake_progress > 0.0 {
            p += 100;
        }
        p
    }
}

pub struct Board {
    width: u32,
    height: u32,
    grid: Vec<Option<Piece>>, // Piece at each cell, indexed as x + y * width
    empty: UVec2,             // The position of the empty cell
    pub hover: Option<UVec2>, // The cell location the cursor is at
    pub reveal: bool,         // Whether to always reveal the kind of piece
}

impl Board {
    pub fn new_shuffle(board_size: BoardSize, rule: Rule) -> Board {
        loop {
            let mut board = Board::new(board_size, rule);
            if board.shuffle() {
                return board;
            }
        }
    }

    fn new(board_size: BoardSize, rule: Rule) -> Board {
        let (width, height) = match board_size {
            BoardSize::Small => (4, 4),
            BoardSize::Large => (8, 8),
        };
        let empty = uvec2(0, 0);
        let grid = (0..height)
            .flat_map(|y| (0..width).map(move |x| UVec2 { x, y }))
            .map(|pos| {
                (pos != empty).then(|| {
                    let kind = match rule {
                        Rule::Classic => Kind::Normal,
                        Rule::Horsey => Kind::Knight,
                        Rule::Full => {
                            let kind_number = thread_rng().gen_range(1..4);
                            match kind_number {
                                1 => Kind::Knight,
                                2 => Kind::Bishop,
                                3 => Kind::Rook,
                                _ => unreachable!(),
                            }
                        }
                    };
                    Piece {
                        kind,
                        source_pos: pos,
                        moving: None,
                        morph_progress: 0.0,
                        shake_progress: 0.0,
                    }
                })
            })
            .collect();
        Board {
            width,
            height,
            grid,
            empty,
            hover: None,
            reveal: false,
        }
    }

    pub fn empty(&self) -> UVec2 {
        self.empty
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn cell(&self, pos: UVec2) -> &Option<Piece> {
        let i = pos.x + pos.y * self.width;
        &self.grid[i as usize]
    }

    pub fn cell_mut(&mut self, pos: UVec2) -> &mut Option<Piece> {
        let i = pos.x + pos.y * self.width;
        &mut self.grid[i as usize]
    }

    fn take(&mut self, pos: UVec2) -> Piece {
        self.cell_mut(pos).take().unwrap()
    }

    fn put(&mut self, pos: UVec2, piece: Piece) {
        *self.cell_mut(pos) = Some(piece);
    }

    // Move the piece at pos to the empty cell
    pub fn move_piece(&mut self, pos: UVec2, no_animation: bool) {
        let mut piece = self.take(pos);
        if no_animation {
            piece.moving = None;
        } else if let Some(moving) = &mut piece.moving {
            // If we are already moving,
            // assuming the existing movement is the inverse movment of the new one.
            // So we can just flip the progress.
            // TODO: the assumption above isn't correct
            // if the player chains multiple on one piece quickly.
            moving.previous_pos = pos;
            moving.progress = 1.0 - moving.progress;
        } else {
            piece.moving = Some(Moving {
                previous_pos: pos,
                progress: 0.0,
            });
        }
        self.put(self.empty, piece);
        self.empty = pos;
    }

    // Test whether the piece at pos can be moved to the empty cell
    pub fn can_move(&mut self, pos: UVec2) -> bool {
        match self.cell(pos).as_ref().unwrap().kind {
            Kind::Normal | Kind::Rook => {
                if pos.x == self.empty.x {
                    pos.y == self.empty.y + 1 || pos.y + 1 == self.empty.y
                } else if pos.y == self.empty.y {
                    pos.x == self.empty.x + 1 || pos.x + 1 == self.empty.x
                } else {
                    false
                }
            }
            Kind::Knight => {
                let rx = pos.x == self.empty.x + 1 || pos.x + 1 == self.empty.x;
                let ry = pos.y == self.empty.y + 1 || pos.y + 1 == self.empty.y;
                let rrx = pos.x == self.empty.x + 2 || pos.x + 2 == self.empty.x;
                let rry = pos.y == self.empty.y + 2 || pos.y + 2 == self.empty.y;
                (rrx && ry) || (rx && rry)
            }
            Kind::Bishop => {
                let rx = pos.x == self.empty.x + 1 || pos.x + 1 == self.empty.x;
                let ry = pos.y == self.empty.y + 1 || pos.y + 1 == self.empty.y;
                rx && ry
            }
        }
    }

    // Shuffle. Returns false if it failed to shuffle with the current piece configuration
    fn shuffle(&mut self) -> bool {
        for _ in 0..1000 {
            let mut options = vec![];
            for dx in -2..=2 {
                for dy in -2..=2 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let pos = self.empty.as_ivec2() + ivec2(dx, dy);
                    if pos.x < 0 || pos.y < 0 {
                        continue;
                    }
                    let pos = pos.as_uvec2();
                    if pos.x >= self.width || pos.y >= self.height {
                        continue;
                    }
                    if !self.can_move(pos) {
                        continue;
                    }
                    options.push(pos);
                }
            }
            if options.is_empty() {
                // "No move" can only happen at the first step.
                // Reject immediately if it happens.
                return false;
            }
            let choice = options[thread_rng().gen_range(0..options.len())];
            self.move_piece(choice, true);
        }

        // Reject the shuffle if too many pieces (> 50%) are at the original position
        let mut unmoved_count = 0;
        for x in 0..self.width {
            for y in 0..self.height {
                let pos = uvec2(x, y);
                if let Some(piece) = self.cell(pos) {
                    if piece.source_pos == pos {
                        unmoved_count += 1;
                    }
                }
            }
        }
        if unmoved_count > self.width * self.height / 2 {
            return false;
        }

        true
    }

    // Start shaking a piece indicating it cannot be moved
    pub fn shake(&mut self, pos: UVec2) {
        self.cell_mut(pos).as_mut().unwrap().shake_progress = 1.0;
    }

    // Move forward animation. Return false if there is no more animation
    pub fn delta(&mut self, delta: Duration) -> bool {
        let mut something_moving = false;
        let delta = delta.as_secs_f32();
        for (i, cell) in self.grid.iter_mut().enumerate() {
            let pos = uvec2(i as u32 % self.width, i as u32 / self.width);

            let Some(piece) = cell else {
                continue;
            };

            if piece.shake_progress > 0.0 {
                something_moving = true;
                piece.shake_progress =
                    (piece.shake_progress - delta * PIECE_SHAKING_SPEED).max(0.0);
            }

            let new_morph_progress = if Some(pos) == self.hover || self.reveal {
                if piece.morph_progress != 1.0 {
                    something_moving = true;
                }
                (piece.morph_progress + delta * PIECE_MORPHING_SPEED).min(1.0)
            } else {
                if piece.morph_progress != 0.0 {
                    something_moving = true;
                }
                (piece.morph_progress - delta * PIECE_MORPHING_SPEED).max(0.0)
            };
            piece.morph_progress = new_morph_progress;

            if let Some(moving) = &mut piece.moving {
                something_moving = true;
                let new_progress = moving.progress + delta * PIECE_MOVING_SPEED;
                if new_progress >= 1.0 {
                    piece.moving = None;
                } else {
                    moving.progress = new_progress;
                }
            }
        }

        something_moving
    }
}
